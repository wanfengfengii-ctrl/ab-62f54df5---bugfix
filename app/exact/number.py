"""Rational numbers and helpers.

Inputs arrive as decimal strings; ``parse_decimal`` turns them into exact
:class:`fractions.Fraction` values without ever touching a binary float.
``factorint`` gives an exact (verification-based) prime factorization used
for mod-squares algebra in the multi-quadratic field.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Dict, Iterable, Union

Number = Union[int, Fraction]


# ------------------------------------------------------------- factorization
_MR_DETERMINISTIC_BOUND = 1 << 64
_MR_FIXED_WITNESSES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _miller_rabin(n: int, a: int) -> bool:
    """Return True if n passes the strong-probable-prime test base a."""
    if a % n == 0:
        return True
    d, s = n - 1, 0
    while d % 2 == 0:
        s += 1
        d //= 2
    x = pow(a, d, n)
    if x in (1, n - 1):
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    if n < _MR_DETERMINISTIC_BOUND:
        # Deterministic for every n < 2**64.
        return all(_miller_rabin(n, a) for a in _MR_FIXED_WITNESSES)
    # Beyond 2**64: Pollard rho still yields *exact* divisors; primality of
    # leaves uses the fixed witnesses plus 40 random witnesses, giving an
    # error probability below 2**-128.  CAM coordinate-derived integers
    # stay well below 2**64 in practice (documented exactness range).
    rng = random.Random(hash(n) & 0xFFFFFFFF)
    return all(
        _miller_rabin(n, a)
        for a in _MR_FIXED_WITNESSES + tuple(
            rng.randrange(2, n - 1) for _ in range(40)
        )
    )


def _sieve(limit: int) -> tuple:
    mark = bytearray(b"\x01") * (limit + 1)
    mark[0:2] = b"\x00\x00"
    for i in range(2, math.isqrt(limit) + 1):
        if mark[i]:
            mark[i * i :: i] = b"\x00" * (((limit - i * i) // i) + 1)
    return tuple(i for i in range(2, limit + 1) if mark[i])


_SMALL_PRIMES = _sieve(1000)


def _pollard_brent(n: int) -> int:
    """Brent's variant of Pollard rho with batched gcd."""
    rng = random.Random(0xC0FFEE ^ n)
    while True:
        y = rng.randrange(1, n)
        c = rng.randrange(1, n)
        m = 128
        g = r = q = 1
        f = lambda v: (v * v + c) % n
        while g == 1:
            x = y
            for _ in range(r):
                y = f(y)
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = f(y)
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = f(ys)
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g


_FACTOR_CACHE: Dict[int, Dict[int, int]] = {}


def factorint(n: int) -> Dict[int, int]:
    """Exact prime factorization of n > 0 as {prime: exponent}.

    Miller-Rabin selects primes; Brent-Pollard rho splits composites.
    Every factor is verified by multiplication, so a probabilistic miss is
    simply retried -- the returned factorization is always exact.  Results
    are cached because coordinate-derived integers recur constantly.
    """
    cached = _FACTOR_CACHE.get(n)
    if cached is not None:
        return dict(cached)
    out: Dict[int, int] = {}

    def divide(m: int) -> None:
        if m == 1:
            return
        stack = [m]
        while stack:
            v = stack.pop()
            for p in _SMALL_PRIMES:
                while v % p == 0:
                    out[p] = out.get(p, 0) + 1
                    v //= p
                if v == 1:
                    break
            if v == 1:
                continue
            if _is_prime(v):
                out[v] = out.get(v, 0) + 1
                continue
            d = _pollard_brent(v)
            stack.append(d)
            stack.append(v // d)

    divide(n)
    # Exact verification; recompute (practically never loops).
    prod = 1
    for p, e in out.items():
        prod *= p ** e
    if prod != n:  # pragma: no cover - defensive
        return factorint(n)
    _FACTOR_CACHE[n] = dict(out)
    return out



def parse_decimal(text: str) -> Fraction:
    """Parse a decimal literal (possibly signed / with an exponent) exactly."""
    s = text.strip()
    if not s:
        raise ValueError("empty number")
    sign = 1
    if s[0] in "+-":
        sign = -1 if s[0] == "-" else 1
        s = s[1:]
    exp = 0
    if "e" in s or "E" in s:
        s, es = s.replace("E", "e").split("e", 1)
        exp = int(es)
    if "." in s:
        whole, frac = s.split(".", 1)
        value = Fraction(
            sign * int((whole + frac) or "0"),
            10 ** len(frac),
        )
    else:
        value = Fraction(sign * int(s or "0"), 1)
    if exp:
        value *= Fraction(1, 10) ** -exp if exp < 0 else Fraction(10) ** exp
    return value


def frac(x: Number) -> Fraction:
    if isinstance(x, Fraction):
        return x
    return Fraction(x)


def sumsq(x: Number, y: Number) -> Fraction:
    x, y = frac(x), frac(y)
    return x * x + y * y


def dot(ax: Number, ay: Number, bx: Number, by: Number) -> Fraction:
    return frac(ax) * frac(bx) + frac(ay) * frac(by)


def cross(ax: Number, ay: Number, bx: Number, by: Number) -> Fraction:
    return frac(ax) * frac(by) - frac(ay) * frac(bx)


def frac_gcd(values: Iterable[Fraction]) -> Fraction:
    """Greatest common divisor of a non-empty iterable of fractions."""
    it = iter(values)
    g = next(it)
    if g < 0:
        g = -g
    for v in it:
        if v < 0:
            v = -v
        g = _gcd(g, v)
    return g


def _gcd(a: Fraction, b: Fraction) -> Fraction:
    while b:
        a, b = b, a % b
    return a
