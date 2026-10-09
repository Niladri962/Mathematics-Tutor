import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adaptive import next_question
from src.data import load_kb
from src.diagnose import Diagnoser
from src.features import expand
from src.llm import LLM
from src.mastery import P_INIT, StudentStore, bkt_update
from src.tutor import Tutor


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def test_bank_is_consistent(kb):
    for q in kb.questions.values():
        assert q["concept_id"] in kb.concepts
        assert q["correct"] in q["options"]
        assert q["tags"].get(q["correct"]) is None, "the correct option must not carry a misconception"
        for m in q["tags"].values():
            assert m is None or m in kb.misconceptions


def test_bkt_moves_in_the_right_direction():
    assert bkt_update(P_INIT, True) > P_INIT
    assert bkt_update(0.9, False) < 0.9


def test_tagged_distractor_is_diagnosed_directly(kb):
    q = kb.questions["Q01"]
    d = Diagnoser(kb).diagnose(q, q["options"]["A"], chosen_option="A")
    assert (d.misconception_id, d.route) == ("M01", "tagged")


def test_free_text_answer_is_diagnosed_by_retrieval(kb):
    # A typed answer arrives with its question, so the topic is known.
    q = dict(kb.questions["Q19"], tags={})
    d = Diagnoser(kb).diagnose(q, "x^2", "x + x = x^2")
    assert d.route == "retrieval"
    assert "M13" in [m for m, _ in d.shortlist[:3]]


def test_off_syllabus_question_is_refused(kb):
    tutor = Tutor(kb, LLM(api_key=""))
    assert tutor.answer("Who won the IPL in 2024?").refused
    reply = tutor.answer("How do I add fractions with different denominators?")
    assert not reply.refused and reply.sources


def test_next_question_targets_the_same_misconception(kb):
    q, why = next_question(kb, {}, {"Q01"}, kb.questions["Q01"], "M01", False, random.Random(0))
    assert "M01" in q["tags"].values() and q["id"] != "Q01"


def test_expansion_detects_fraction_addition():
    assert "adding fractions" in expand("3/4 + 1/2 = ?")


def test_misconception_cleared_after_targeted_success(kb, tmp_path):
    store = StudentStore(tmp_path / "t.db")
    store.record("s", kb.questions["Q01"], "A", "M01", ts=1)
    store.record("s", kb.questions["Q02"], "B", None, ts=2)
    row = store.misconception_status().iloc[0]
    assert row.status == "cleared" and row.evidence == "possible"
