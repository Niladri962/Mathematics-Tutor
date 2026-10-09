"""Evaluate the tutor.

1. Diagnosis. For every wrong option tagged with a misconception, hide the tag and
   ask the retrieval route to diagnose from the question text and the chosen answer
   alone. Reported: top-1 accuracy, top-3 accuracy, and MAP@25 (the Eedi competition
   metric). Run in two settings: topic known (realistic inside a practice session)
   and topic unknown (a student typing any wrong answer).
2. Refusal. In-scope questions should be answered; off-syllabus ones declined.

Usage:  python -m eval.evaluate          (add --llm to include the LLM re-ranker)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import PROCESSED, load_kb  # noqa: E402
from src.diagnose import Diagnoser  # noqa: E402
from src.llm import LLM  # noqa: E402
from src.tutor import Tutor  # noqa: E402

IN_SCOPE = [  # (question, concept the top source should come from)
    ("Why is 0.5 bigger than 0.125?", "C04"),
    ("How do I add fractions with different denominators?", "C01"),
    ("What does a power of zero mean?", "C11"),
    ("How do I expand a bracket with a minus sign in front?", "C09"),
    ("Why does dividing by a half make the answer bigger?", "C03"),
    ("How do I share money in a ratio?", "C13"),
    ("What is the rule for BODMAS?", "C07"),
    ("How do I solve 4y + 3 = 19?", "C10"),
    ("Can I add 3x and 2y together?", "C08"),
    ("Why is -4 + -6 not 10?", "C06"),
]
OUT_OF_SCOPE = [
    "Who won the IPL in 2024?",
    "Write me an essay on the Mughal empire",
    "What is the capital of Australia?",
    "Tell me a joke about cricket",
    "Explain photosynthesis",
    "Translate good morning into French",
]


def diagnosis_eval(kb, diag, llm=None, k=25):
    test_ids = None
    split = PROCESSED / "eedi_test_ids.json"
    if split.exists():
        test_ids = set(json.loads(split.read_text()))
    cases = []
    for q in kb.questions.values():
        if test_ids is not None and q["id"] not in test_ids:
            continue
        for opt, mid in q.get("tags", {}).items():
            if mid and opt != q["correct"]:
                cases.append((q, opt, mid))
    out = {}
    for setting in ("topic_known", "topic_unknown"):
        top1 = top3 = ap = 0.0
        misses = []
        for q, opt, mid in cases:
            query = diag.build_query(q["text"], q["options"][opt])
            cid = q["concept_id"] if setting == "topic_known" else None
            ranked = [m for m, _ in diag.rank(query, cid, k=k)]
            if llm is not None and llm.available:
                pick = llm.rerank_misconception(q["text"], q["options"][opt], "",
                                                [kb.misconception(m) for m in ranked[:5]])
                if pick:
                    ranked = [pick] + [m for m in ranked if m != pick]
            if ranked and ranked[0] == mid:
                top1 += 1
            else:
                misses.append({"question": q["id"], "answer": q["options"][opt], "expected": mid,
                               "got": ranked[0] if ranked else None})
            top3 += mid in ranked[:3]
            ap += 1 / (ranked.index(mid) + 1) if mid in ranked else 0
        n = len(cases)
        out[setting] = {"cases": n, "top1": round(top1 / n, 3), "top3": round(top3 / n, 3),
                        "map@25": round(ap / n, 3), "misses": misses}
    return out


def ask_eval(tutor, kb):
    replies = [(tutor.answer(q), cid) for q, cid in IN_SCOPE]
    answered = sum(not r.refused for r, _ in replies)
    right_source = sum(bool(r.sources) and r.sources[0].title == kb.concept(cid)["name"] for r, cid in replies)
    refused = sum(tutor.answer(q).refused for q in OUT_OF_SCOPE)
    return {"in_scope_answered": f"{answered}/{len(IN_SCOPE)}",
            "top_source_correct": f"{right_source}/{len(IN_SCOPE)}",
            "out_of_scope_refused": f"{refused}/{len(OUT_OF_SCOPE)}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="include the LLM re-ranker")
    args = ap.parse_args()
    kb = load_kb()
    llm = LLM() if args.llm else None
    diag = Diagnoser(kb)
    tutor = Tutor(kb, LLM(api_key=""))  # refusal is decided by retrieval, so test it offline
    results = {"diagnosis": diagnosis_eval(kb, diag, llm), "ask_mode": ask_eval(tutor, kb),
               "data": "eedi" if (PROCESSED / "eedi_test_ids.json").exists() else "seed"}
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Data: {results['data']}  |  LLM re-ranker: {'on' if llm and llm.available else 'off'}")
    print(f"{'Setting':<15}{'Cases':>7}{'Top-1':>8}{'Top-3':>8}{'MAP@25':>9}")
    for s, r in results["diagnosis"].items():
        print(f"{s:<15}{r['cases']:>7}{r['top1']:>8.1%}{r['top3']:>8.1%}{r['map@25']:>9.3f}")
    print("Ask mode:", results["ask_mode"])
    print(f"Full results, including misses, in {out}")


if __name__ == "__main__":
    main()
