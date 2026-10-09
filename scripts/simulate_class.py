"""Populate the database with a simulated class so the teacher view has data to show.

Each simulated student is given two or three hidden misconceptions. When a question
has a distractor for one of them, the student picks it 80% of the time. Otherwise they
answer correctly with a probability set by their ability. Because the planted
misconceptions are known, you can check that the dashboard surfaces the right ones.

Usage:  python -m scripts.simulate_class --students 12 --questions 18
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adaptive import next_question  # noqa: E402
from src.data import ROOT, load_kb  # noqa: E402
from src.diagnose import Diagnoser  # noqa: E402
from src.mastery import StudentStore  # noqa: E402

NAMES = ["Aarav", "Ishita", "Kabir", "Meera", "Rohan", "Sana", "Vihaan", "Ananya",
         "Arjun", "Diya", "Kiaan", "Nisha", "Om", "Pari", "Reyansh", "Tara"]


def simulate(n_students=12, n_questions=24, seed=11, reset=True):
    rng = random.Random(seed)
    kb, store = load_kb(), StudentStore()
    diag = Diagnoser(kb)
    if reset:
        store.reset()
    truth = {}
    t0 = time.time() - 7 * 86400
    for s_i, name in enumerate(NAMES[:n_students]):
        planted = rng.sample(list(kb.misconceptions), rng.choice([2, 3]))
        ability = rng.uniform(0.55, 0.9)
        truth[name] = planted
        seen, last_q, last_m, last_ok = set(), None, None, True
        for step in range(n_questions):
            q, _ = next_question(kb, store.mastery(name), seen, last_q, last_m, last_ok, rng)
            if q is None:
                break
            hit = [opt for opt, m in q["tags"].items() if m in planted]
            if hit and rng.random() < 0.8:
                chosen = hit[0]
            elif rng.random() < ability:
                chosen = q["correct"]
            else:
                chosen = rng.choice([o for o in q["options"] if o != q["correct"]])
            d = diag.diagnose(q, q["options"][chosen], chosen_option=chosen)
            store.record(name, q, chosen, d.misconception_id, ts=t0 + s_i * 600 + step * 30)
            seen.add(q["id"])
            last_q, last_m, last_ok = q, d.misconception_id, chosen == q["correct"]
    (ROOT / "data" / "simulated_truth.json").write_text(json.dumps(truth, indent=1))
    return truth, store


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=12)
    ap.add_argument("--questions", type=int, default=24)
    a = ap.parse_args()
    truth, store = simulate(a.students, a.questions)
    status, att = store.misconception_status(), store.attempts()
    kb = load_kb()
    exposed = [(s, m) for s, ms in truth.items() for m in ms
               if any(m in str(t).split(",") for t in att[att.student == s].targets)]
    found = [(s, m) for s, m in exposed
             if ((status.student == s) & (status.misconception_id == m)).any()]
    conf = status[status.evidence == "confirmed"]
    true_conf = sum(m in truth[s] for s, m in zip(conf.student, conf.misconception_id))
    print(f"Simulated {len(truth)} students, {len(att)} attempts.")
    print(f"Recall: {len(found)}/{len(exposed)} planted misconceptions surfaced, "
          f"among those the student was actually tested on.")
    print(f"Precision of 'confirmed' flags: {true_conf}/{len(conf)}. "
          f"'Possible' flags (single wrong answer): {(status.evidence == 'possible').sum()}.")
