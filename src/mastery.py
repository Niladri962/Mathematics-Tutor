"""Student model.

Mastery per concept is tracked with Bayesian Knowledge Tracing (Corbett and Anderson,
1994): after each attempt the probability that the student knows the concept is
updated for slips and guesses, then for the chance of learning on that step.

A misconception is "open" once diagnosed and "cleared" when the student later answers
correctly a question that has a distractor targeting that same misconception.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

from .data import ROOT

DB_PATH = ROOT / "data" / "tutor.db"
P_INIT, P_LEARN, P_SLIP, P_GUESS = 0.30, 0.15, 0.10, 0.25

SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  student TEXT, question_id TEXT, concept_id TEXT, chosen TEXT,
  correct INTEGER, misconception_id TEXT, targets TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS mastery (
  student TEXT, concept_id TEXT, p REAL, PRIMARY KEY (student, concept_id));
"""


def bkt_update(p: float, correct: bool) -> float:
    if correct:
        post = p * (1 - P_SLIP) / (p * (1 - P_SLIP) + (1 - p) * P_GUESS)
    else:
        post = p * P_SLIP / (p * P_SLIP + (1 - p) * (1 - P_GUESS))
    return post + (1 - post) * P_LEARN


class StudentStore:
    def __init__(self, path: Path | str = DB_PATH):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def mastery(self, student: str) -> dict:
        rows = self.conn.execute("SELECT concept_id, p FROM mastery WHERE student=?", (student,)).fetchall()
        return dict(rows)

    def record(self, student, question, chosen, misconception_id=None, ts=None) -> float:
        correct = chosen == question["correct"]
        targets = ",".join(sorted({t for t in question.get("tags", {}).values() if t}))
        self.conn.execute(
            "INSERT INTO attempts (student, question_id, concept_id, chosen, correct, misconception_id, targets, ts)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (student, question["id"], question["concept_id"], chosen, int(correct),
             misconception_id, targets, ts or time.time()))
        p = self.mastery(student).get(question["concept_id"], P_INIT)
        p_new = bkt_update(p, correct)
        self.conn.execute("INSERT OR REPLACE INTO mastery VALUES (?,?,?)", (student, question["concept_id"], p_new))
        self.conn.commit()
        return p_new

    def seen(self, student) -> set:
        return {r[0] for r in self.conn.execute("SELECT question_id FROM attempts WHERE student=?", (student,))}

    def attempts(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM attempts ORDER BY ts", self.conn)

    def students(self) -> list:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT student FROM attempts ORDER BY student")]

    def mastery_matrix(self) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM mastery", self.conn)
        if df.empty:
            return df
        return df.pivot(index="student", columns="concept_id", values="p")

    def misconception_status(self) -> pd.DataFrame:
        """One row per (student, misconception).

        evidence: "confirmed" when the student made this error at least twice AND on at
        least half of the questions that tested for it (a consistent pattern, not a
        slip); otherwise "possible".
        status: "cleared" if a later question targeting it was answered correctly.
        """
        df = self.attempts()
        cols = ["student", "misconception_id", "times", "opportunities", "rate", "evidence", "status"]
        if df.empty:
            return pd.DataFrame(columns=cols)
        df["target_list"] = df.targets.fillna("").str.split(",")
        rows = []
        for (student, mid), g in df[df.misconception_id.notna()].groupby(["student", "misconception_id"]):
            mine = df[df.student == student]
            tested = mine[mine.target_list.apply(lambda t: mid in t)]
            opportunities = max(len(tested), len(g))
            rate = len(g) / opportunities
            later_ok = tested[(tested.ts > g.ts.min()) & (tested.correct == 1)]
            rows.append({"student": student, "misconception_id": mid, "times": len(g),
                         "opportunities": opportunities, "rate": round(rate, 2),
                         "evidence": "confirmed" if len(g) >= 2 and rate >= 0.5 else "possible",
                         "status": "cleared" if len(later_ok) else "open"})
        return pd.DataFrame(rows, columns=cols)

    def reset(self):
        self.conn.executescript("DELETE FROM attempts; DELETE FROM mastery;")
        self.conn.commit()
