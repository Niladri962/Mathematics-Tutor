"""Choose the next question.

Priority order:
  1. After a diagnosed misconception: an unseen question with a distractor that targets
     the same misconception. Answering it correctly is evidence the gap has closed.
  2. After a wrong answer without a diagnosis: another unseen question on that concept.
  3. Otherwise: an unseen question from the concept with the lowest mastery estimate.
"""
from __future__ import annotations

import random

from .data import KnowledgeBase
from .mastery import P_INIT


def next_question(kb: KnowledgeBase, mastery: dict, seen: set, last_question: dict | None = None,
                  last_misconception: str | None = None, last_correct: bool = True, rng=random):
    unseen = [q for q in kb.questions.values() if q["id"] not in seen]
    if not unseen:
        return None, "You have answered every question in the bank."
    if last_misconception:
        targeted = [q for q in unseen if last_misconception in q.get("tags", {}).values()]
        if targeted:
            name = kb.misconception(last_misconception)["name"].lower()
            return rng.choice(targeted), f"Checking the same idea again: {name}."
    if last_question and not last_correct:
        same = [q for q in unseen if q["concept_id"] == last_question["concept_id"]]
        if same:
            return rng.choice(same), "Another one on the same topic."
    by_concept = {}
    for q in unseen:
        by_concept.setdefault(q["concept_id"], []).append(q)
    weakest = min(by_concept, key=lambda c: (mastery.get(c, P_INIT), rng.random()))
    return rng.choice(by_concept[weakest]), f"Next: {kb.concept(weakest)['name'].lower()}, your weakest topic so far."
