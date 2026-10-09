"""A small, safe calculator for the Ask tab.

Students often type a plain sum ("4567 x 987", "3/4 + 1/2", "7/0", "20% of 150").
This module recognises those and answers them exactly.

Safety: nothing here uses eval() or exec(). The text is first checked against a strict
whitelist of characters (digits, + - * / ( ) . ^ and x), then parsed with Python's
`ast` module, and only number, sign, + - * / and whole-number powers are allowed.
Anything else (letters, function calls, names) is rejected before it is evaluated.
Length, number size and exponent size are capped so a request cannot hang the app.

Arithmetic uses fractions.Fraction, so 0.1 + 0.2 is exactly 0.3 and 1/3 stays 1/3.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

MAX_LEN = 100          # characters in the expression
MAX_NUMBER_LEN = 15    # characters in any one number typed by the student
MAX_EXPONENT = 200     # largest whole-number power allowed
MAX_BITS = 4000        # size cap on any intermediate or final result

LEAD_IN = re.compile(
    r"^(?:please\s+)?(?:what\s*(?:is|'s|s)|how\s+much\s+is|calculate|compute|evaluate|"
    r"work\s+out|find|solve|simplify)\s*(?:the\s+(?:value|answer|result)\s+of\s*)?",
    re.I,
)
PERCENT_OF = re.compile(r"^(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)$", re.I)
ALLOWED = re.compile(r"^[\d\s+\-*/().]+$")


@dataclass
class CalcResult:
    text: str      # the reply sentence
    lookup: str    # the cleaned expression, used to find a related topic
    ok: bool       # False when undefined or refused


class _Refuse(Exception):
    pass


def _clean(query: str) -> str:
    s = query.strip().rstrip("?=! ").strip()
    s = LEAD_IN.sub("", s, count=1).strip()
    s = re.sub(r"\s*(?:equals?|=)\s*$", "", s, flags=re.I).rstrip("?= ").strip()
    s = re.sub(r"(?<=\d),(?=\d{2,3}(?!\d))", "", s)           # 4,567 -> 4567
    s = s.replace("×", "*").replace("÷", "/").replace("^", "**")
    s = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(.])", "*", s)     # 3 x 4 -> 3*4
    return s


def _display(s: str) -> str:
    s = re.sub(r"\s+", " ", s).replace("**", "^").replace("*", "×")
    return re.sub(r"\s*×\s*", " × ", s)


def _group(n: int) -> str:
    """Indian digit grouping: 4507629 -> 45,07,629."""
    sign, digits = ("-" if n < 0 else ""), str(abs(n))
    if len(digits) <= 3:
        return sign + digits
    head, tail = digits[:-3], digits[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return sign + ",".join(parts + [tail])


def _terminates(x: Fraction) -> bool:
    d = x.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return d == 1


def _decimal(x: Fraction) -> str:
    with localcontext() as ctx:
        ctx.prec = 60
        dec = Decimal(x.numerator) / Decimal(x.denominator)
        if _terminates(x):
            return format(dec.normalize(), "f")
        return format(dec.quantize(Decimal("0.000001")).normalize(), "f")


def _eval(node) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        v = node.value
        return Fraction(v) if isinstance(v, int) else Fraction(repr(v))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Add):
            r = a + b
        elif isinstance(node.op, ast.Sub):
            r = a - b
        elif isinstance(node.op, ast.Mult):
            r = a * b
        elif isinstance(node.op, ast.Div):
            if b == 0:
                raise ZeroDivisionError
            r = a / b
        elif isinstance(node.op, ast.Pow):
            if b.denominator != 1:
                raise _Refuse("I can only do whole-number powers.")
            if abs(b) > MAX_EXPONENT:
                raise _Refuse("That power is too large for me to work out.")
            if a == 0 and b < 0:
                raise ZeroDivisionError
            r = a ** int(b)
        else:
            raise _Refuse("I can only do + - × ÷ and powers.")
        if r.numerator.bit_length() > MAX_BITS or r.denominator.bit_length() > MAX_BITS:
            raise _Refuse("That number is too large for me to work out.")
        return r
    raise _Refuse("I can only do plain arithmetic.")


def try_calculate(query: str) -> CalcResult | None:
    """Return a CalcResult if the query is plain arithmetic, otherwise None."""
    s = _clean(query)
    if not s or len(s) > MAX_LEN:
        return None

    m = PERCENT_OF.match(s)
    if m:
        a, b = Fraction(m.group(1)), Fraction(m.group(2))
        r = a / 100 * b
        return CalcResult(f"{m.group(1)}% of {m.group(2)} = {_decimal(r) if r.denominator != 1 else _group(int(r))}",
                          s, True)

    if not ALLOWED.match(s) or not re.search(r"\d", s):
        return None
    if not re.search(r"[+\-*/]", s.lstrip("-+( ")):     # a bare number is not a sum
        return None
    shown = _display(s)
    lookup = s.replace("**", "^")          # the form the topic rules in features.py recognise
    if any(len(n) > MAX_NUMBER_LEN for n in re.findall(r"[\d.]+", s)):
        return CalcResult("That number is too long for me. Try a smaller one.", lookup, False)
    try:
        r = _eval(ast.parse(s, mode="eval"))
    except ZeroDivisionError:
        return CalcResult(
            f"{shown} is undefined, because it divides by zero. Dividing asks how many "
            "zeros fit into a number, and no amount of zeros ever adds up to anything. "
            "Another way to see it: if 7 / 0 had an answer k, then k x 0 would have to "
            "equal 7, but anything times 0 is 0.", lookup, False)
    except _Refuse as e:
        return CalcResult(str(e), lookup, False)
    except (SyntaxError, ValueError, RecursionError):
        return CalcResult(f"I couldn't read \"{shown}\" as a sum. Check the brackets and signs.", lookup, False)

    if r.denominator == 1:
        n = int(r)
        digits = len(str(abs(n)))
        if digits > 25:
            lead = str(abs(n))[:4]
            return CalcResult(f"{shown} is a number with {digits} digits, starting {lead[0]}.{lead[1:]} × 10^{digits - 1}.",
                              s, True)
        return CalcResult(f"{shown} = {_group(n)}", lookup, True)
    dec = _decimal(r)
    sign = "=" if _terminates(r) else "≈"
    if "." in s and _terminates(r):
        return CalcResult(f"{shown} = {dec}", lookup, True)
    if shown.replace(" ", "") == f"{r.numerator}/{r.denominator}":      # a lone fraction: just give the decimal
        return CalcResult(f"{shown} {sign} {dec}", lookup, True)
    return CalcResult(f"{shown} = {r.numerator}/{r.denominator} ({sign} {dec})", lookup, True)
