"""Exact planar geometry: points, lines, circles, circular arcs.

All coordinates are :class:`~app.exact.algebraic.Element` values in a shared
:class:`~app.exact.algebraic.Field`.  Every predicate (parallel, tangent,
inside-arc, ordering along an arc) is an exact sign decision; no epsilon is
used anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .algebraic import Element, Field


class Vec:
    __slots__ = ("x", "y")

    def __init__(self, x: Element, y: Element):
        self.x = x
        self.y = y

    def __repr__(self) -> str:  # pragma: no cover - debugging
        return f"Vec({self.x!r}, {self.y!r})"


# ---------------------------------------------------------------- vectors
def vadd(F: Field, a: Vec, b: Vec) -> Vec:
    return Vec(F.add(a.x, b.x), F.add(a.y, b.y))


def vsub(F: Field, a: Vec, b: Vec) -> Vec:
    return Vec(F.sub(a.x, b.x), F.sub(a.y, b.y))


def vscale(F: Field, a: Vec, k) -> Vec:
    k = k if isinstance(k, Element) else F.rational(k)
    return Vec(F.mul(a.x, k), F.mul(a.y, k))


def vdot(F: Field, a: Vec, b: Vec) -> Element:
    return F.add(F.mul(a.x, b.x), F.mul(a.y, b.y))


def vcross(F: Field, a: Vec, b: Vec) -> Element:
    return F.sub(F.mul(a.x, b.y), F.mul(a.y, b.x))


def vlen2(F: Field, a: Vec) -> Element:
    return vdot(F, a, a)


def vunit(F: Field, a: Vec) -> Vec:
    return vscale(F, a, F.inv(F.square_root(vlen2(F, a))))


def veq(F: Field, a: Vec, b: Vec) -> bool:
    return F.eq(a.x, b.x) and F.eq(a.y, b.y)


def left_of(F: Field, v: Vec) -> Vec:
    return Vec(F.neg(v.y), v.x)


def right_of(F: Field, v: Vec) -> Vec:
    return Vec(v.y, F.neg(v.x))


# ----------------------------------------------------------------- curves
@dataclass
class Line:
    """Infinite oriented line p + t*u with unit direction u."""

    p: Vec
    u: Vec


@dataclass
class Circle:
    c: Vec
    r: Element  # strictly positive


# ------------------------------------------------------------ intersections
Intersection = Tuple[Vec, str]  # (point, kind: 'cross' | 'tangent' | 'overlap')


def line_line_intersection(
    F: Field, l1: Line, l2: Line
) -> Tuple[Optional[Element], Optional[Vec], bool]:
    """Return (t1, point, coincident).  Parallel distinct -> (None, None, False)."""
    delta = vsub(F, l2.p, l1.p)
    det = vcross(F, l1.u, l2.u)
    if F.is_zero(det):
        if F.is_zero(vcross(F, l1.u, delta)):
            return None, None, True
        return None, None, False
    t = F.div(vcross(F, delta, l2.u), det)
    return t, Vec(F.add(l1.p.x, F.mul(t, l1.u.x)), F.add(l1.p.y, F.mul(t, l1.u.y))), False


def line_circle_intersections(F: Field, l: Line, cir: Circle) -> List[Tuple[Element, Vec, str]]:
    """Full-line / circle intersections as (t, point, kind)."""
    pc = vsub(F, l.p, cir.c)
    b = vdot(F, pc, l.u)
    cc = F.sub(vlen2(F, pc), F.sq(cir.r))
    disc = F.sub(F.sq(b), cc)
    s = F.sign(disc)
    if s < 0:
        return []
    t0 = F.neg(b)
    if s == 0:
        p = Vec(F.add(l.p.x, F.mul(t0, l.u.x)), F.add(l.p.y, F.mul(t0, l.u.y)))
        return [(t0, p, "tangent")]
    h = F.square_root(disc)

    def pt(t: Element) -> Vec:
        return Vec(F.add(l.p.x, F.mul(t, l.u.x)), F.add(l.p.y, F.mul(t, l.u.y)))

    tm, tp = F.sub(t0, h), F.add(t0, h)
    return [(tm, pt(tm), "cross"), (tp, pt(tp), "cross")]


def circle_circle_intersections(
    F: Field, c1: Circle, c2: Circle
) -> Tuple[List[Tuple[Vec, str]], str]:
    """Return ([(point, kind), ...], relation).

    relation is one of: 'cross', 'tangent_external', 'tangent_internal',
    'disjoint', 'nested', 'coincident'.
    """
    dvec = vsub(F, c2.c, c1.c)
    d2 = vlen2(F, dvec)
    if F.is_zero(d2):
        if F.eq(c1.r, c2.r):
            return [], "coincident"
        return [], "nested"
    rsum = F.add(c1.r, c2.r)
    rdiff = F.sub(c1.r, c2.r)
    sum2 = F.sq(rsum)
    diff2 = F.sq(rdiff)
    if F.gt(d2, sum2):
        return [], "disjoint"
    if F.lt(d2, diff2):
        return [], "nested"
    d = F.square_root(d2)
    a = F.div(F.add(F.sub(F.sq(c1.r), F.sq(c2.r)), d2), F.add(d, d))
    # base = c1 + (a/d) * dvec
    e = vscale(F, dvec, F.div(a, d))
    base = vadd(F, c1.c, e)
    if F.eq(d2, sum2) or F.eq(d2, diff2):
        return [(base, "tangent")], (
            "tangent_external" if F.eq(d2, sum2) else "tangent_internal"
        )
    h2 = F.sub(F.sq(c1.r), F.sq(a))
    h = F.square_root(h2)
    perp = Vec(F.neg(dvec.y), dvec.x)  # length d
    off = vscale(F, perp, F.div(h, d))
    return [
        (vadd(F, base, off), "cross"),
        (vsub(F, base, off), "cross"),
    ], "cross"
