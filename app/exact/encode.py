"""Serialization of exact field elements.

Every algebraic result is rendered two ways:

* ``exact``: a recomputable nested expression.  An element of
  ``Q(s_0, ..., s_{m-1})`` is the sum of its sparse basis terms::

      {"sum": [{"rational": [n, d]},
               {"*": [{"monomial": [i, j]}, {"sqrt": <D_i rec.>}, ...]}]}

  so a downstream system can re-parse it and reproduce the value exactly;
* ``decimal``: a high-precision Decimal display value (never used for any
  geometric decision).
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Any, Dict

from .algebraic import Element, Field


def _quantize(text_fmt, places: int) -> str:
    q = Decimal(1).scaleb(-places)
    return format(text_fmt.quantize(q), "f")


def radicand_exact(F: Field, index: int, memo: Dict[int, Any]) -> Any:
    if index not in memo:
        memo[index] = encode_exact(F, F.radicands[index], memo)
    return memo[index]


def encode_exact(F: Field, x: Element, memo: Dict[int, Any] = None) -> Any:
    """Structural exact form of a sparse multi-quadratic element."""
    if memo is None:
        memo = {}
    terms = []
    for mono, c in sorted(x.terms.items(), key=lambda kv: (len(kv[0]), kv[0])):
        if not mono:
            terms.append(_rational(c))
            continue
        factors = []
        for i in mono:
            factors.append({"sqrt": radicand_exact(F, i, memo)})
        if c == 1 and len(factors) == 1:
            terms.append(factors[0])
        elif c == 1:
            factors.sort(key=lambda _: 0)
            terms.append({"*": factors})
        else:
            factors.insert(0, _rational(c))
            terms.append({"*": factors})
    if not terms:
        return {"rational": [0, 1]}
    if len(terms) == 1:
        return terms[0]
    return {"sum": terms}


def _rational(c: Fraction) -> Dict[str, Any]:
    return {"rational": [c.numerator, c.denominator]}


# ----------------------------------------------------------- decimal display
def evaluate_decimal(F: Field, x: Element, places: int = 12) -> Decimal:
    """Compute a high-precision decimal by exact recursive sqrt evaluation.

    The working precision is fixed once from ``places``; callers must not
    feed the current context precision back in (that would grow without
    bound).
    """
    getcontext().prec = max(places + 12, 40)
    memo: Dict[int, Decimal] = {}
    return _eval(F, x, memo)


def _rad_decimal(F: Field, i: int, memo: Dict[int, Decimal]) -> Decimal:
    if i not in memo:
        memo[i] = _eval(F, F.radicands[i], memo).sqrt()
    return memo[i]


def _eval(F: Field, x: Element, memo: Dict[int, Decimal]) -> Decimal:
    total = Decimal(0)
    for mono, c in x.terms.items():
        val = Decimal(c.numerator) / Decimal(c.denominator)
        for i in mono:
            val *= _rad_decimal(F, i, memo)
        total += val
    return total


def encode_number(F: Field, x: Element, places: int = 12) -> Dict[str, Any]:
    dec = evaluate_decimal(F, x, places)
    return {
        "exact": encode_exact(F, x),
        "decimal": _quantize(dec, places),
        "sign": F.sign(x),
    }


def encode_point(F: Field, p, places: int = 12) -> Dict[str, Any]:
    return {
        "x": encode_number(F, p.x, places),
        "y": encode_number(F, p.y, places),
    }
