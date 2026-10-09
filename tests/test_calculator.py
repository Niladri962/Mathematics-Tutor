import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calculator import try_calculate
from src.data import load_kb
from src.llm import LLM
from src.tutor import Tutor, chunk_lesson
from src.retriever import HybridIndex


@pytest.mark.parametrize("query, expected", [
    ("what is 4567 * 987", "45,07,629"),
    ("4567 x 987", "45,07,629"),
    ("3/4 + 1/2", "5/4 (= 1.25)"),
    ("0.1 + 0.2", "0.3"),                 # exact, unlike floating point
    ("12 x 0.4", "4.8"),
    ("9 - (-2)", "11"),
    ("-4 + (-6)", "-10"),
    ("2^10", "1,024"),
    ("what is 20% of 150", "30"),
    ("1/3", "≈ 0.333333"),
])
def test_correct_answers(query, expected):
    r = try_calculate(query)
    assert r is not None and r.ok and expected in r.text


def test_division_by_zero_is_explained_not_crashed():
    for q in ("what is 7/0?", "0/0", "5/(3-3)"):
        r = try_calculate(q)
        assert r is not None and not r.ok and "undefined" in r.text


@pytest.mark.parametrize("query", [
    "__import__('os').system('dir')",      # code injection
    "open('secrets.txt')",
    "how do I solve 5y + 4 = 29?",          # algebra, not arithmetic
    "Why is 0.5 bigger than 0.125?",
    "2x + 3",
    "what is 5",                            # a bare number is not a sum
])
def test_non_arithmetic_is_not_intercepted(query):
    assert try_calculate(query) is None


@pytest.mark.parametrize("query", ["9**9**9", "2^1000", "2^0.5", "1234567890123456 + 1", "(3+4", "5 +"])
def test_unsafe_or_malformed_input_is_refused_safely(query):
    r = try_calculate(query)
    assert r is not None and not r.ok


def test_answer_links_to_the_idea_behind_the_sum():
    tutor = Tutor(load_kb(), LLM(api_key=""))
    reply = tutor.answer("what is 9 - (-2)?")
    assert "= 11" in reply.text and reply.sources and "integers" in reply.sources[0].title.lower()
    plain = tutor.answer("what is 4567 * 987")
    assert "45,07,629" in plain.text and not plain.sources


def test_calculator_works_in_lesson_mode_too():
    tutor = Tutor(load_kb(), LLM(api_key=""))
    lesson = "Part 1: Integers\n\nSubtracting a negative number is the same as adding its opposite. For 9 - (-2) we get 11."
    idx = HybridIndex(chunk_lesson(lesson), lambda d: f"{d['title']}. {d['text']}")
    reply = tutor.answer("what is 9 - (-2)?", lesson_index=idx)
    assert "= 11" in reply.text
