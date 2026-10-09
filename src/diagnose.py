"""Work out *why* a student gave a wrong answer.

Two routes:
  1. Tagged route. If the question is multiple choice and the chosen wrong option is
     tagged with a misconception, that tag is the diagnosis (confidence 1.0).
  2. Retrieval route. For free-text answers, or untagged distractors, the question,
     the student's answer and their working are used as a query against the
     misconception library. Concept retrieval acts as a prior, so a misconception
     from the right topic is preferred. An optional LLM re-ranker can reorder the
     shortlist.
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import KnowledgeBase
from .features import expand
from .retriever import HybridIndex


@dataclass
class Diagnosis:
    misconception_id: str | None
    confidence: float
    route: str                  # "tagged", "retrieval", "retrieval+llm" or "none"
    shortlist: list[tuple[str, float]]


def _misc_text(m):
    return f"{m['name']}. {m['description']} {m['signature']}"


def _concept_text(c, miscs):
    sigs = " ".join(m["signature"] for m in miscs if m["concept_id"] == c["id"])
    return f"{c['name']}. {c['strand']}. {c['note']} {sigs}"


class Diagnoser:
    def __init__(self, kb: KnowledgeBase, concept_weight: float = 0.6):
        self.kb = kb
        self.concept_weight = concept_weight
        miscs = list(kb.misconceptions.values())
        self.misc_index = HybridIndex(miscs, _misc_text)
        self.concept_index = HybridIndex(list(kb.concepts.values()), lambda c: _concept_text(c, miscs))

    @staticmethod
    def build_query(question_text: str, answer: str, working: str = "", expanded: bool = False) -> str:
        parts = [expand(question_text) if expanded else question_text, f"student answer {answer}"]
        if working:
            parts.append(f"working {working}")
        return ". ".join(parts)

    def rank(self, query: str, concept_id: str | None = None, k: int = 5, concept_query: str | None = None):
        n_concepts = len(self.kb.concepts)
        if concept_id:
            prior = {concept_id: 1.0}
        else:
            hits = self.concept_index.search(concept_query or expand(query), k=n_concepts)
            top = hits[0].score
            prior = {h.id: h.score / top for h in hits}
        # With the topic unknown, structural expansion helps; with it known, it adds noise.
        misc_query = query if concept_id else (concept_query or expand(query))
        hits = self.misc_index.search(misc_query, k=len(self.kb.misconceptions))
        top = hits[0].score
        scored = []
        for h in hits:
            c = h.payload["concept_id"]
            s = (1 - self.concept_weight) * (h.score / top) + self.concept_weight * prior.get(c, 0.0)
            scored.append((h.id, round(s, 4)))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def diagnose(self, question: dict | None, answer: str, working: str = "",
                 chosen_option: str | None = None, llm=None) -> Diagnosis:
        if question and chosen_option:
            if chosen_option == question.get("correct"):
                return Diagnosis(None, 1.0, "none", [])
            tag = question.get("tags", {}).get(chosen_option)
            if tag:
                return Diagnosis(tag, 1.0, "tagged", [(tag, 1.0)])
        q_text = question["text"] if question else ""
        concept_id = question["concept_id"] if question else None
        shortlist = self.rank(self.build_query(q_text, answer, working), concept_id)
        route = "retrieval"
        if llm is not None and llm.available:
            choice = llm.rerank_misconception(q_text, answer, working,
                                              [self.kb.misconception(m) for m, _ in shortlist])
            if choice:
                shortlist = [(choice, 1.0)] + [s for s in shortlist if s[0] != choice]
                route = "retrieval+llm"
        best, score = shortlist[0]
        second = shortlist[1][1] if len(shortlist) > 1 else 0.0
        confidence = round(min(1.0, 0.5 + (score - second)), 2)
        return Diagnosis(best, confidence, route, shortlist)
