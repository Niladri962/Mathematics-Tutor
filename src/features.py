"""Query expansion from the structure of a maths expression.

A question like "3/4 + 1/2 = ?" contains almost no words, so lexical retrieval cannot
tell which topic it belongs to. These rules read the expression's structure and append
topic vocabulary to the query. They are deliberately simple and transparent.
"""
from __future__ import annotations

import re

FRAC = r"\d+/\d+"
DEC = r"\d*\.\d+"
RULES = [
    (rf"{FRAC}\s*[+\-]\s*{FRAC}", "adding fractions common denominator numerator"),
    (rf"{FRAC}\s*[x*]\s*{FRAC}", "multiplying fractions product multiply straight across"),
    (rf"({FRAC}|{DEC}|\d+)\s*(divided by|÷)\s*({FRAC}|{DEC})", "dividing fractions division reciprocal smaller bigger"),
    (rf"\d+\s*[x*]\s*{DEC}|{DEC}\s*[x*]\s*\d", "multiplying decimals decimal places product"),
    (rf"(largest|smallest|bigger|smaller|greater|less)\D*{DEC}|{DEC}\D*(largest|smallest|bigger|smaller)", "comparing decimals tenths place value"),
    (r"^which\b.*(largest|smallest|bigger|smaller)", "comparing decimals tenths place value"),
    (r"\bdivid(e|es|ing|ed)\b", "dividing fractions division reciprocal"),
    (r"\(\s*-\s*\d|^\s*-\d+\s*[+\-]", "negative numbers integers number line adding subtracting"),
    (r"\bsolve\b|[a-z]\s*[+\-]\s*\d+\s*=|\d[a-z]\s*=", "solving linear equation balance both sides"),
    (r"\^\s*\d|\bpower\b|squared", "exponents power base"),
    (r"\)\s*\^\s*2", "identity square of a sum middle term 2ab"),
    (r"%|percent", "percentage change original value"),
    (r"\bratio\b|\d+\s*:\s*\d+", "ratio share parts total"),
    (r"\bexpand\b|\d\s*\(|-\s*\(", "expanding brackets multiply every term"),
    (r"\bsimplify\b|\b\d*[a-z]\s*[+\-]\s*\d*[a-z]\b", "like terms unlike terms collecting algebra"),
    (r"^\s*\d+\s*[+\-]\s*\d+\s*[x*/]\s*\d+|^\s*\d+\s*[x*/]\s*\d+\s*[+\-]\s*\d+", "order of operations BODMAS precedence"),
]


def expand(text: str) -> str:
    low = text.lower()
    extra = [words for pattern, words in RULES if re.search(pattern, low)]
    return f"{text} {' '.join(extra)}" if extra else text
