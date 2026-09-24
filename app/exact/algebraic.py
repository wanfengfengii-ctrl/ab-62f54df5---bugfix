"""Exact real algebraic numbers in a multi-quadratic field.

The ambient field is the multi-quadratic extension

    Q(s_1, ..., s_m),   s_i**2 = D_i,

where each positive radicand ``D_i`` is itself an element of
``Q(s_1, ..., s_{i-1})``.  Because every extension degree is two, the
multiplication rule closes on the subset monomials

    e_S = prod_{i in S} s_i        (S a sorted tuple of variable indices)

with ``e_S * e_T = (prod_{i in S&T} D_i) * e_{S ^ T}``.  An element is a
*sparse* dictionary ``{S: rational coefficient}``; geometric quantities only
ever involve a handful of radicals, so the dictionaries stay tiny and the
cost never grows with the total number of radicals adjoined during a job.

Sign decisions use the field axioms only -- never floating point: writing
``x = a + b s_m`` for the highest variable ``m`` appearing in x,

* equal signs of a, b decide x immediately;
* opposite signs are resolved by the exact norm ``a**2 - b**2 D_m``,

recursing until a single rational coefficient (the monomial bases are all
strictly positive under the chosen embedding ``s_i = +sqrt(D_i)``).
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .number import factorint, frac


_SF_CACHE: Dict[Tuple[int, int], tuple] = {}


def squarefree_key(q: Fraction) -> tuple:
    """Odd-exponent prime signature of a positive rational (tuple, sorted).

    Two positive rationals differ by a rational square iff their keys are
    equal.  Factorization is exact (see :func:`app.exact.number.factorint`).

    This classification is used only to *find* candidate roots; every
    candidate is subsequently verified by the exact identity ``y**2 == x``,
    so a classification mistake can never produce a wrong result -- at worst
    it forces a redundant tower extension (never an incorrect one).
    """
    key = (q.numerator, q.denominator)
    cached = _SF_CACHE.get(key)
    if cached is not None:
        return cached
    fac: Dict[int, int] = {}
    for n, sign in ((q.numerator, 1), (q.denominator, -1)):
        for p, e in factorint(n).items():
            if (e * sign) & 1:
                fac[p] = fac.get(p, 0) ^ 1
    result = tuple(sorted(p for p, bit in fac.items() if bit))
    _SF_CACHE[key] = result
    return result

# A basis monomial is a sorted tuple of adjoined-variable indices.
Monomial = Tuple[int, ...]
EMPTY: Monomial = ()


def _symdiff(a: Monomial, b: Monomial) -> Monomial:
    return tuple(sorted(set(a).symmetric_difference(set(b))))


def _inter(a: Monomial, b: Monomial) -> Tuple[int, ...]:
    return tuple(sorted(set(a).intersection(b)))


class Element:
    """Sparse element: sum of coeff[S] * e_S."""

    __slots__ = ("terms",)

    def __init__(self, terms: Optional[Dict[Monomial, Fraction]] = None):
        self.terms = dict(terms) if terms else {}

    # Convenience used only in debugging / tests.
    @property
    def k(self) -> int:
        """Highest variable index + 1 (0 for rationals)."""
        top = 0
        for s in self.terms:
            if s and s[-1] + 1 > top:
                top = s[-1] + 1
        return top

    @property
    def a(self):
        """Rational part when the element is a bare rational."""
        return self.terms.get(EMPTY, Fraction(0))

    @property
    def b(self):
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging
        return f"E({self.terms})"


class Field:
    """Multi-quadratic field with sparse rational-basis arithmetic."""

    def __init__(self):
        # Positive radicands; radicands[i] is an Element using only
        # variables with index < i.
        self.radicands: List[Element] = []
        # F2 Gaussian basis for square classes of *rational* radicands:
        # reduced squarefree key -> (variable indices producing it).
        self._sf_pivot: Dict[tuple, Tuple[int, ...]] = {}
        # Exact monomial product cache (see _monomial_product).
        self._monoprod_cache: Dict[
            Tuple[Tuple[int, ...], Tuple[int, ...]], Element
        ] = {}

    # ------------------------------------------------------------- basics
    @property
    def level(self) -> int:
        return len(self.radicands)

    def rational(self, value) -> Element:
        v = value if isinstance(value, Fraction) else frac(value)
        if v == 0:
            return Element()
        return Element({EMPTY: v})

    zero = lambda self, k=None: Element()
    one = lambda self, k=None: Element({EMPTY: Fraction(1)})

    def align(self, x: Element, y: Element) -> Tuple[Element, Element]:
        return x, y

    def reduce(self, x: Element) -> Element:
        return x

    # ----------------------------------------------------------- arithmetic
    @staticmethod
    def _put(terms: Dict[Monomial, Fraction], s: Monomial, v: Fraction) -> None:
        if v == 0:
            terms.pop(s, None)
        else:
            terms[s] = terms.get(s, Fraction(0)) + v
            if terms[s] == 0:
                del terms[s]

    def add(self, x: Element, y: Element) -> Element:
        terms = dict(x.terms)
        for s, c in y.terms.items():
            self._put(terms, s, c)
        return Element(terms)

    def sub(self, x: Element, y: Element) -> Element:
        terms = dict(x.terms)
        for s, c in y.terms.items():
            self._put(terms, s, -c)
        return Element(terms)

    def neg(self, x: Element) -> Element:
        return Element({s: -c for s, c in x.terms.items()})

    def _times_variable(self, x: Element, j: int) -> Element:
        """x * s_j via the defining relation: (a + b*s_j)*s_j = a*s_j + b*D_j."""
        terms: Dict[Monomial, Fraction] = {}
        for s, c in x.terms.items():
            if j in s:
                reduced = tuple(i for i in s if i != j)
                piece = self.scale(self.radicands[j], c)
                part = self._times_monomial(piece, reduced)
                for sd, cd in part.terms.items():
                    self._put(terms, sd, cd)
            else:
                self._put(terms, tuple(sorted(set(s) | {j})), c)
        return Element(terms)

    def _times_monomial(self, x: Element, mono: Tuple[int, ...]) -> Element:
        out = x
        for j in sorted(mono, reverse=True):
            out = self._times_variable(out, j)
        return out

    def _monomial_product(
        self, m1: Tuple[int, ...], m2: Tuple[int, ...]
    ) -> Element:
        """Memoised e_m1 * e_m2, computed by exact defining relations."""
        key = (m1, m2) if m1 <= m2 else (m2, m1)
        cached = self._monoprod_cache.get(key)
        if cached is not None:
            return cached
        value = self._times_monomial(Element({m1: Fraction(1)}), m2)
        self._monoprod_cache[key] = value
        return value

    def mul(self, x: Element, y: Element) -> Element:
        """Sparse bilinear multiplication with memoised monomial products."""
        if not x.terms or not y.terms:
            return Element()
        terms: Dict[Monomial, Fraction] = {}
        for sx, cx in x.terms.items():
            for sy, cy in y.terms.items():
                prod = self._monomial_product(sx, sy)
                coef = cx * cy
                for s, c in prod.terms.items():
                    self._put(terms, s, coef * c)
        return Element(terms)

    def scale(self, x: Element, n) -> Element:
        n = n if isinstance(n, Fraction) else frac(n)
        return Element({s: c * n for s, c in x.terms.items() if c * n != 0})

    def sq(self, x: Element) -> Element:
        return self.mul(x, x)

    def square(self, x: Element) -> Element:
        return self.sq(x)

    # ------------------------------------------------------------- queries
    def is_zero(self, x: Element) -> bool:
        return not x.terms

    def eq(self, x: Element, y: Element) -> bool:
        return self.is_zero(self.sub(x, y))

    def sign(self, x: Element) -> int:
        """Exact sign under s_i = +sqrt(D_i), via nested norms."""
        terms = x.terms
        if not terms:
            return 0
        if len(terms) == 1:
            (c,) = terms.values()
            return (c > 0) - (c < 0)
        m = self._highest_var(x)
        a_t: Dict[Monomial, Fraction] = {}
        b_t: Dict[Monomial, Fraction] = {}
        for s, c in terms.items():
            if m in s:
                b_t[tuple(i for i in s if i != m)] = c
            else:
                a_t[s] = c
        a, b = Element(a_t), Element(b_t)
        sa, sb = self.sign(a), self.sign(b)
        if sb == 0:
            return sa
        if sa == 0:
            return sb  # s_m is positive
        if sa == sb:
            return sa
        n = self.sub(self.sq(a), self.mul(self.sq(b), self.radicands[m]))
        sn = self.sign(n)
        if sn == 0:
            return 0
        return sa * sn

    @staticmethod
    def _highest_var(x: Element) -> int:
        m = -1
        for s in x.terms:
            if s and s[-1] > m:
                m = s[-1]
        return m

    def lt(self, x, y) -> bool:
        return self.sign(self.sub(x, y)) < 0

    def le(self, x, y) -> bool:
        return self.sign(self.sub(x, y)) <= 0

    def gt(self, x, y) -> bool:
        return self.sign(self.sub(x, y)) > 0

    def ge(self, x, y) -> bool:
        return self.sign(self.sub(x, y)) >= 0

    # ------------------------------------------------------------ inversion
    def inv(self, x: Element) -> Element:
        if self.is_zero(x):
            raise ZeroDivisionError
        used = {i for s in x.terms for i in s}
        if not used:
            c = x.terms[EMPTY]
            return Element({EMPTY: Fraction(1) / c})
        # Tower recursion: 1/(a + b s_m) = (a - b s_m) / (a**2 - b**2 D_m).
        m = max(used)
        norm_m = self.mul(x, self._conjugate(x, m))  # lies below level m
        return self.mul(self._conjugate(x, m), self.inv(norm_m))

    def _conjugate(self, x: Element, m: int) -> Element:
        return Element(
            {s: (-c if m in s else c) for s, c in x.terms.items()}
        )

    def div(self, x: Element, y: Element) -> Element:
        return self.mul(x, self.inv(y))

    # -------------------------------------------------------- square roots
    def _rational_square_root(self, a: Fraction) -> Optional[Fraction]:
        ra, rb = math.isqrt(a.numerator), math.isqrt(a.denominator)
        if ra * ra == a.numerator and rb * rb == a.denominator:
            return Fraction(ra, rb)
        return None

    def _square_via_subsets(self, x: Element) -> Optional[Element]:
        """Root of a rational x using the F2 basis of radicand classes.

        If x = c**2 * prod_{i in M} D_i then sqrt(x) = c * e_M; the F2
        reduction finds M in polynomial time.
        """
        if len(x.terms) != 1 or EMPTY not in x.terms:
            return None
        q = x.terms[EMPTY]
        residual, mono = self._reduce_rational_class(q)
        if residual:
            return None
        prod = self._rational_monomial_product(tuple(sorted(mono)))
        c = self._rational_square_root(q / prod)
        if c is None:
            return None
        if c == 0:
            return Element()
        return Element({tuple(sorted(mono)): c})

    def sqrt_exact(self, x: Element, _below: Optional[int] = None) -> Optional[Element]:
        """Positive y with y**2 == x, or None if x is not a square here.

        When ``_below`` is given the answer is required to lie in the
        subfield spanned by variables with index < ``_below``.  This is
        mandatory inside the tower decomposition below: the norm root and
        the two half-roots must belong to the subfield under s_m.  An
        ambient-field root that itself involves s_m (e.g. a rational whose
        square class only becomes a square through s_m) must NOT be accepted
        there -- otherwise p2 = (A + delta)/2 still contains s_m and the
        recursion never descends the tower.
        """
        sx = self.sign(x)
        if sx < 0:
            return None
        if sx == 0:
            return Element()
        if _below is not None and any(
            (s and s[-1] >= _below) for s in x.terms
        ):
            return None
        used = {i for s in x.terms for i in s}
        if not used:
            a = x.terms[EMPTY]
            rr = self._rational_square_root(a)
            r = Element({EMPTY: rr}) if rr is not None else self._square_via_subsets(x)
            if r is not None and _below is not None and any(
                (s and s[-1] >= _below) for s in r.terms
            ):
                return None
            return r
        m = max(used)
        a_t: Dict[Monomial, Fraction] = {}
        b_t: Dict[Monomial, Fraction] = {}
        for s, c in x.terms.items():
            if m in s:
                b_t[tuple(i for i in s if i != m)] = c
            else:
                a_t[s] = c
        A, B = Element(a_t), Element(b_t)
        Dm = self.radicands[m]
        if not b_t:
            r = self.sqrt_exact(A, _below=m)
            if r is not None:
                return r
            q = self.sqrt_exact(self.mul(A, self.inv(Dm)), _below=m)
            if q is not None:
                return self._with_var(q, m)
            return None
        delta2 = self.sub(self.sq(A), self.mul(self.sq(B), Dm))
        # delta = sqrt(N_{m}(x)) must be an element of the subfield below m;
        # an answer that needs s_m means x has no square root in this field.
        root = self.sqrt_exact(delta2, _below=m)
        if root is None:
            return None
        half = self.rational(Fraction(1, 2))
        two = self.rational(2)
        for sign in (1, -1):
            delta = root if sign == 1 else self.neg(root)
            p2 = self.mul(self.add(A, delta), half)
            q2 = self.mul(self.sub(A, delta), self.inv(self.mul(two, Dm)))
            if self.sign(p2) < 0 or self.sign(q2) < 0:
                continue
            # p and q are coefficients in the (1, s_m) decomposition and so
            # must live strictly below m.
            p = self.sqrt_exact(p2, _below=m)
            q = self.sqrt_exact(q2, _below=m)
            if p is None or q is None:
                continue
            # Enforce 2 p q == B by flipping q's sign when necessary.
            pq2 = self.mul(two, self.mul(p, q))
            if (self.sign(pq2) > 0) != (self.sign(B) > 0):
                q = self.neg(q)
            y = self.add(p, self._with_var(q, m))
            if self.eq(self.sq(y), x):
                return y if self.sign(y) >= 0 else self.neg(y)
        return None

    @staticmethod
    def _with_var(x: Element, m: int) -> Element:
        return Element(
            {tuple(sorted(set(s) | {m})): c for s, c in x.terms.items()}
        )

    def square_root(self, x: Element) -> Element:
        """Positive square root; adjoins a new variable only when the
        radicand is genuinely new (not a square multiple of an existing
        one).

        Canonicalization is essential: re-deriving the same geometric
        intersection must reuse the same algebraic variable, otherwise two
        equal real numbers would carry independent representations and fail
        exact equality.
        """
        if self.sign(x) < 0:
            raise ValueError("negative radicand")
        r = self.sqrt_exact(x)
        if r is not None:
            return r
        # x = q**2 * D_i  =>  sqrt(x) = q * s_i.
        for i in range(self.level):
            q = self.sqrt_exact(self.mul(x, self.inv(self.radicands[i])))
            if q is not None:
                return self.mul(q, Element({(i,): Fraction(1)}))
        self.extend(x)
        return Element({(self.level - 1,): Fraction(1)})

    def extend(self, radicand) -> int:
        d = radicand if isinstance(radicand, Element) else self.rational(radicand)
        if self.sign(d) <= 0:
            raise ValueError("only positive radicands can be adjoined")
        index = self.level
        self.radicands.append(d)
        self._register_rational_class(index, d)
        return self.level

    def _register_rational_class(self, index: int, d: Element) -> None:
        """Register the mod-squares class of a rational radicand.

        Gaussian elimination over F2 on prime parities lets square roots of
        rationals that are products of radicands be found in polynomial time
        instead of enumerating 2**m subsets.
        """
        if len(d.terms) != 1 or EMPTY not in d.terms:
            return  # non-rational radicands never yield rational monomials
        key = set(squarefree_key(d.terms[EMPTY]))
        mono = {index}
        # Eliminate pivot primes already in the span.
        changed = True
        while changed:
            changed = False
            for p in tuple(key):
                hit = self._sf_pivot.get(p)
                if hit is None:
                    continue
                pivot_key, pivot_mono = hit
                key.symmetric_difference_update(pivot_key)
                mono.symmetric_difference_update(pivot_mono)
                changed = True
                break
        if key:
            head = min(key)
            self._sf_pivot[head] = (frozenset(key), frozenset(mono))

    def _reduce_rational_class(self, q: Fraction):
        """Reduce q's squarefree class against the F2 basis.

        Returns (residual prime set, monomial indices M) with
        q = (rational square) * prod_{i in M} D_i after reduction.  An empty
        residual means q is a square times that monomial product.
        """
        key = set(squarefree_key(q))
        mono: set = set()
        changed = True
        while changed:
            changed = False
            for p in tuple(key):
                hit = self._sf_pivot.get(p)
                if hit is None:
                    continue
                pivot_key, pivot_mono = hit
                key.symmetric_difference_update(pivot_key)
                mono.symmetric_difference_update(pivot_mono)
                changed = True
                break
        return key, mono

    def _rational_monomial_product(self, mono) -> Fraction:
        out = Fraction(1)
        for i in mono:
            out *= self.radicands[i].terms[EMPTY]
        return out

    # ------------------------------------------------------------ helpers
    def norm(self, x: Element) -> Element:
        """Full field norm down to Q (a rational element)."""
        cur = x
        for m in range(self.level - 1, -1, -1):
            if any(m in s for s in cur.terms):
                cur = self.mul(cur, self._conjugate(cur, m))
        return cur
