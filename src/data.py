"""Load the knowledge base: concepts, misconceptions, questions and optional NCERT chunks.

Seed data lives in data/seed/. If you run the ingestion scripts, processed files in
data/processed/ are picked up automatically and take priority over the seed files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "data" / "seed"
PROCESSED = ROOT / "data" / "processed"


def _load(name: str):
    processed = PROCESSED / name
    path = processed if processed.exists() else SEED / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class KnowledgeBase:
    concepts: dict
    misconceptions: dict
    questions: dict
    ncert_chunks: list = field(default_factory=list)

    def concept(self, cid):
        return self.concepts.get(cid)

    def misconception(self, mid):
        return self.misconceptions.get(mid)

    def questions_for(self, concept_id):
        return [q for q in self.questions.values() if q["concept_id"] == concept_id]


def load_kb() -> KnowledgeBase:
    concepts = {c["id"]: c for c in _load("concepts.json")}
    miscs = {m["id"]: m for m in _load("misconceptions.json")}
    questions = {q["id"]: q for q in _load("questions.json")}
    chunks = []
    chunk_file = PROCESSED / "ncert_chunks.json"
    if chunk_file.exists():
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
    return KnowledgeBase(concepts, miscs, questions, chunks)
