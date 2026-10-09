"""Turn a diagnosis into a reply, and answer open questions from the knowledge base.

Every reply carries its sources. Ask mode refuses when retrieval finds nothing close
enough, so the tutor does not drift into answering from general knowledge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .calculator import CalcResult, try_calculate
from .data import KnowledgeBase
from .features import expand
from .llm import LLM, TUTOR_SYSTEM
from .retriever import HybridIndex

REFUSAL_KEYWORD = 1.0      # top BM25 score below this means no shared subject vocabulary
REFUSAL_LEXICAL = 0.10     # and a weak character-level match as a second check
LANGS = {"English": "English", "Hindi": "Hindi (Devanagari script)", "Marathi": "Marathi (Devanagari script)"}


@dataclass
class Source:
    label: str
    title: str
    text: str
    ref: str


@dataclass
class Reply:
    text: str
    sources: list = field(default_factory=list)
    refused: bool = False
    used_llm: bool = False


def concept_source(kb: KnowledgeBase, cid: str, label="S1") -> Source:
    c = kb.concept(cid)
    return Source(label, c["name"], c["note"], f"NCERT {c['ncert_topic']}")


def chunk_lesson(text: str, max_words: int = 110) -> list[dict]:
    """Split a lesson into passages that follow its own structure.

    Paragraphs are kept whole, and each passage carries the nearest heading above it
    (a short line without a full stop, such as "Part 7: Expanding brackets"), so
    retrieval knows the topic and answers do not start mid-sentence. Paragraphs longer
    than max_words are split at sentence boundaries. A transcript with no paragraph
    breaks is split the same way.
    """
    chunks, heading = [], "Today's lesson"

    def add(body: str):
        chunks.append({"id": f"L{len(chunks) + 1}", "title": heading, "text": body})

    for block in re.split(r"\n\s*\n", text):
        block = " ".join(block.split())
        if not block:
            continue
        if len(block.split()) <= 12 and not block.endswith((".", "?", "!")):
            heading = block
            continue
        current: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            if current and len(" ".join(current + [sentence]).split()) > max_words:
                add(" ".join(current))
                current = []
            current.append(sentence)
        if current:
            add(" ".join(current))
    return chunks


class Tutor:
    def __init__(self, kb: KnowledgeBase, llm: LLM | None = None):
        self.kb = kb
        self.llm = llm or LLM()
        docs = [{"id": c["id"], "kind": "concept", "title": c["name"], "text": c["note"],
                 "ref": f"NCERT {c['ncert_topic']}"} for c in kb.concepts.values()]
        docs += [{"id": ch["id"], "kind": "ncert", "title": ch.get("title", "NCERT"),
                  "text": ch["text"], "ref": ch.get("ref", "NCERT")} for ch in kb.ncert_chunks]
        self.book_index = HybridIndex(docs, lambda d: f"{d['title']}. {d['text']}")

    # ---- Practise mode -------------------------------------------------
    def feedback(self, question: dict, chosen: str, misconception_id: str | None,
                 language: str = "English") -> Reply:
        correct = chosen == question["correct"]
        src = concept_source(self.kb, question["concept_id"])
        if correct:
            return Reply("Correct. Ready for the next one?", [src])
        m = self.kb.misconception(misconception_id) if misconception_id else None
        if self.llm.available:
            diag = (f"Likely misconception: {m['name']}. {m['description']}" if m
                    else "No specific misconception identified; treat as a slip and re-check the method.")
            user = (f"Sources:\n[S1] {src.title} ({src.ref}): {src.text}\n\n"
                    f"Question: {question['text']}\nStudent chose: {question['options'][chosen]}\n"
                    f"Correct answer: {question['options'][question['correct']]}\n{diag}\n"
                    "Write the reply: one line naming the likely thinking, one guiding question, "
                    "then a short grounded explanation. Do not give the final answer outright.")
            out = self.llm.complete(TUTOR_SYSTEM.format(language=LANGS.get(language, "English")), user)
            if out:
                return Reply(out, [src], used_llm=True)
        if m:
            text = (f"It looks like you may be thinking: {m['description'][0].lower()}{m['description'][1:]}\n"
                    f"Think about a similar case: {m['probe']}\n"
                    f"Remember: {m['remedy']} [S1]")
        else:
            text = f"Not quite. Re-check your method against the rule: {src.text.split('. ')[0]}. [S1]"
        return Reply(text, [src])

    # ---- Ask mode -------------------------------------------------------
    def _calc_reply(self, calc: CalcResult, lesson_index: HybridIndex | None) -> Reply:
        """Answer a plain sum exactly, then point to the idea behind it when there is one."""
        if not calc.ok:
            return Reply(calc.text)
        recognised = expand(calc.lookup)
        if recognised == calc.lookup:           # no known pattern (e.g. a plain product): no topic to link
            return Reply(calc.text)
        index = lesson_index or self.book_index
        hits = index.search(recognised, k=3)
        # Lessons explain a method before they warn about its mistakes, so among
        # near-equal passages prefer the one that comes first in the lesson.
        close = [h for h in hits if h.score >= 0.9 * hits[0].score]
        hit = min(close, key=lambda h: int(re.sub(r"\D", "", str(h.id)) or 0)) if lesson_index else hits[0]
        if hit.keyword < REFUSAL_KEYWORD or hit.lexical < REFUSAL_LEXICAL:
            return Reply(calc.text)
        d = hit.payload
        src = Source("S1", d.get("title", "Course material"), d["text"], d.get("ref", "Today's lesson"))
        return Reply(f"{calc.text}\n\nHow it works. From {src.title}: {src.text} [S1]", [src])

    def answer(self, query: str, language: str = "English", lesson_index: HybridIndex | None = None,
               k: int = 3) -> Reply:
        calc = try_calculate(query)
        if calc:
            return self._calc_reply(calc, lesson_index)
        index = lesson_index or self.book_index
        hits = index.search(query if lesson_index else expand(query), k=k)
        best_kw = max((h.keyword for h in hits), default=0.0)
        if not hits or best_kw < REFUSAL_KEYWORD or hits[0].lexical < REFUSAL_LEXICAL:
            return Reply("I can only help with the maths topics in the course material. "
                         "Try rephrasing, or ask your teacher about this one.", refused=True)
        kept = [h for h in hits if h.keyword > 0 or h is hits[0]]
        sources = []
        for i, h in enumerate(kept):
            d = h.payload
            title = d.get("title", f"Lesson extract {d['id']}")
            ref = d.get("ref", "Today's lesson")
            sources.append(Source(f"S{i + 1}", title, d["text"], ref))
        if self.llm.available:
            ctx = "\n".join(f"[{s.label}] {s.title} ({s.ref}): {s.text}" for s in sources)
            out = self.llm.complete(TUTOR_SYSTEM.format(language=LANGS.get(language, "English")),
                                    f"Sources:\n{ctx}\n\nStudent question: {query}")
            if out:
                return Reply(out, sources, used_llm=True)
        # Offline there is no model to choose between passages, so when two score within
        # 3% of each other (a near-tie) show both rather than gamble on one.
        shown = [src for src, h in zip(sources, kept) if h.score >= 0.97 * kept[0].score][:2]
        text = "\n\n".join(f"From {x.title}: {x.text} [{x.label}]" for x in shown)
        return Reply(text, shown)
