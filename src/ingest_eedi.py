"""Convert the Eedi "Mining Misconceptions in Mathematics" Kaggle dataset to this
project's schema, so the tutor and evaluation run on thousands of real, expert-labelled
distractors instead of the small seed set.

Usage:
    1. Download from https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data
       and place train.csv and misconception_mapping.csv in data/raw/eedi/
    2. python -m src.ingest_eedi --test-size 0.2

Leakage control: misconception "signatures" (the text retrieval matches against) are
built only from training questions. Test question ids are written to
data/processed/eedi_test_ids.json and the evaluation script uses only those.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict

import pandas as pd

from .data import PROCESSED, ROOT

RAW = ROOT / "data" / "raw" / "eedi"
LETTERS = "ABCD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    try:
        train = pd.read_csv(RAW / "train.csv")
        mapping = pd.read_csv(RAW / "misconception_mapping.csv")
    except FileNotFoundError:
        sys.exit(f"Place train.csv and misconception_mapping.csv in {RAW}")

    ids = list(train.QuestionId)
    random.Random(args.seed).shuffle(ids)
    test_ids = set(ids[: int(len(ids) * args.test_size)])

    concepts, questions = {}, []
    sig_parts = defaultdict(list)
    misc_subjects = defaultdict(Counter)
    for _, r in train.iterrows():
        cid = f"E{int(r.SubjectId)}"
        concepts.setdefault(cid, {"id": cid, "name": r.SubjectName, "strand": r.SubjectName,
                                  "ncert_topic": f"mapped topic: {r.SubjectName}",
                                  "note": r.ConstructName})
        tags = {}
        for L in LETTERS:
            m = r.get(f"Misconception{L}Id")
            tags[L] = None if pd.isna(m) else f"EM{int(m)}"
            if tags[L]:
                misc_subjects[tags[L]][cid] += 1
                if r.QuestionId not in test_ids:
                    sig_parts[tags[L]].append(f"{r.QuestionText} answer {r[f'Answer{L}Text']}")
        questions.append({"id": f"EQ{int(r.QuestionId)}", "concept_id": cid, "text": r.QuestionText,
                          "options": {L: str(r[f"Answer{L}Text"]) for L in LETTERS},
                          "correct": r.CorrectAnswer, "tags": tags})

    miscs = []
    for _, r in mapping.iterrows():
        mid = f"EM{int(r.MisconceptionId)}"
        subj = misc_subjects[mid].most_common(1)
        miscs.append({"id": mid, "concept_id": subj[0][0] if subj else next(iter(concepts)),
                      "name": r.MisconceptionName, "description": r.MisconceptionName,
                      "signature": " ; ".join(sig_parts[mid][:5]),
                      "probe": "Can you explain each step of your working out loud?",
                      "remedy": "Compare your method with the worked example in the source."})

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, obj in [("concepts.json", list(concepts.values())), ("misconceptions.json", miscs),
                      ("questions.json", questions), ("eedi_test_ids.json", sorted(f"EQ{i}" for i in test_ids))]:
        (PROCESSED / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(questions)} questions, {len(miscs)} misconceptions, {len(concepts)} topics; "
          f"{len(test_ids)} questions held out for evaluation.")


if __name__ == "__main__":
    main()
