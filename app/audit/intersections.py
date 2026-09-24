"""Exact intersections of *trimmed* chain pieces.

Pieces are closed segments and closed arcs.  Every routine returns exact
field points plus an exact contact classification; coincident overlaps are
reported as intervals with exact endpoints.  No tolerances anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..exact.algebraic import Element, Field
from ..exact.arcs import (
    Arc,
    on_closed_arc,
    same_radius_point,
    CCW,
)
from ..exact.geometry import (
    Circle,
    Vec,
    circle_circle_intersections,
    line_circle_intersections,
    vcross,
    vdot,
    vsub,
)
from .primitives import ArcP, Seg, as_exact_arc


# --------------------------------------------------------------- membership
def seg_contains(F: Field, seg: Seg, p: Vec) -> bool:
    """p lies on the closed segment (known to be on its supporting line)."""
    pa, pb = vsub(F, p, seg.a), vsub(F, p, seg.b)
    ab = vsub(F, seg.b, seg.a)
    return F.sign(vdot(F, pa, ab)) >= 0 and F.sign(vdot(F, pb, ab)) <= 0


def seg_param_cmp(F: Field, seg: Seg, p: Vec, q: Vec) -> int:
    """Order two points on the segment along a -> b."""
    ab = vsub(F, seg.b, seg.a)
    d = F.sub(vdot(F, vsub(F, q, seg.a), ab), vdot(F, vsub(F, p, seg.a), ab))
    s = F.sign(d)
    return (s > 0) - (s < 0)


def arc_contains(F: Field, arc: ArcP, p: Vec) -> bool:
    return on_closed_arc(F, as_exact_arc(F, arc), p)


def piece_contains(F: Field, piece, p: Vec) -> bool:
    if isinstance(piece, Seg):
        return F.is_zero(vcross(F, piece.u, vsub(F, p, piece.a))) and seg_contains(
            F, piece, p
        )
    d = vsub(F, p, piece.c)
    if not F.eq(F.add(F.sq(d.x), F.sq(d.y)), F.sq(piece.r)):
        return False
    return arc_contains(F, piece, p)


def same_point(F: Field, a: Vec, b: Vec) -> bool:
    return F.eq(a.x, b.x) and F.eq(a.y, b.y)


def piece_position(F: Field, piece, p: Vec) -> int:
    """0 at start, 2 at end, 1 in the interior (p known to lie on piece)."""
    if same_point(F, p, piece.start):
        return 0
    if same_point(F, p, piece.end):
        return 2
    return 1


# ------------------------------------------------------------- intersections
@dataclass
class Contact:
    point: Optional[Vec]  # None for interval overlaps
    kind: str  # 'cross' | 'tangent' | 'overlap'
    interval: Optional[Tuple[Vec, Vec]] = None
    pos_a: int = 1
    pos_b: int = 1
    full: bool = False


def seg_seg_contacts(F: Field, a: Seg, b: Seg) -> List[Contact]:
    u, v = a.u, b.u
    w = vsub(F, b.a, a.a)
    det = vcross(F, u, v)
    if not F.is_zero(det):
        # Unique line intersection; keep it if inside both closed segments.
        t = F.div(vcross(F, w, v), det)
        p = Vec(
            F.add(a.a.x, F.mul(t, u.x)),
            F.add(a.a.y, F.mul(t, u.y)),
        )
        if _t_in_segment(F, a, p) and _t_in_segment(F, b, p):
            return [Contact(p, "cross", pos_a=piece_position(F, a, p),
                            pos_b=piece_position(F, b, p))]
        return []
    # Parallel.
    if not F.is_zero(vcross(F, u, w)):
        return []
    # Collinear: overlap of the two closed parameter intervals.
    # Project onto u (unit): scalar s = dot(p - a.a, u).
    def sp(q):
        return vdot(F, vsub(F, q, a.a), u)

    sa0, sa1 = F.rational(0), sp(a.b)
    sb0, sb1 = sp(b.a), sp(b.b)
    lo = max2(F, min2(F, sa0, sa1), min2(F, sb0, sb1))
    hi = min2(F, max2(F, sa0, sa1), max2(F, sb0, sb1))
    if F.lt(hi, lo):
        return []
    plo = Vec(F.add(a.a.x, F.mul(lo, u.x)), F.add(a.a.y, F.mul(lo, u.y)))
    phi = Vec(F.add(a.a.x, F.mul(hi, u.x)), F.add(a.a.y, F.mul(hi, u.y)))
    if F.eq(lo, hi):
        return [Contact(plo, "tangent", pos_a=piece_position(F, a, plo),
                        pos_b=piece_position(F, b, plo))]
    return [Contact(None, "overlap", interval=(plo, phi))]


def _t_in_segment(F: Field, seg: Seg, p: Vec) -> bool:
    return seg_contains(F, seg, p)


def seg_arc_contacts(F: Field, seg: Seg, arc: ArcP) -> List[Contact]:
    from ..exact.geometry import Line

    ln = Line(seg.a, seg.u)
    raw = line_circle_intersections(F, ln, arc.circle)
    ea = as_exact_arc(F, arc)
    out: List[Contact] = []
    for _t, p, kind in raw:
        if not seg_contains(F, seg, p) or not on_closed_arc(F, ea, p):
            continue
        # Tangency requires both the line to be tangent to the circle AND the
        # contact not to be an arc-endpoint crossing geometry; circle kind is
        # 'tangent' exactly when discriminant == 0.
        out.append(
            Contact(p, kind, pos_a=piece_position(F, seg, p),
                    pos_b=piece_position(F, arc, p))
        )
    return out


def arc_arc_contacts(F: Field, a: ArcP, b: ArcP) -> List[Contact]:
    pts, rel = circle_circle_intersections(F, a.circle, b.circle)
    ea, eb = as_exact_arc(F, a), as_exact_arc(F, b)
    if rel == "coincident":
        return _coincident_arc_overlap(F, a, b)
    out: List[Contact] = []
    for p, kind in pts:
        if on_closed_arc(F, ea, p) and on_closed_arc(F, eb, p):
            out.append(
                Contact(p, "tangent" if kind == "tangent" else "cross",
                        pos_a=piece_position(F, a, p),
                        pos_b=piece_position(F, b, p))
            )
    return out


def _coincident_arc_overlap(F: Field, a: ArcP, b: ArcP) -> List[Contact]:
    """Overlap of two closed arcs on one circle, as exact angular intervals.

    The intersection of two connected circle arcs is a (possibly empty)
    collection of at most two closed arc intervals, plus isolated endpoint
    contacts.  Everything is decided by exact on-arc predicates and exact
    CCW ordering; no angular numeric value is ever formed.
    """
    if a.full and b.full:
        return [Contact(None, "overlap", interval=(a.start, b.start), full=True)]
    if a.full:
        return [Contact(None, "overlap", interval=(b.start, b.end),
                        full=False)]
    if b.full:
        return [Contact(None, "overlap", interval=(a.start, a.end),
                        full=False)]
    s_a, e_a = (a.start, a.end) if a.sense == CCW else (a.end, a.start)
    s_b, e_b = (b.start, b.end) if b.sense == CCW else (b.end, b.start)
    arc_a = Arc(a.circle, vsub(F, s_a, a.c), vsub(F, e_a, a.c), CCW)
    arc_b = Arc(b.circle, vsub(F, s_b, b.c), vsub(F, e_b, b.c), CCW)

    # Distinct boundary points in exact CCW order starting at s_a.
    bounds = [s_a, e_a, s_b, e_b]
    ordered: List[Vec] = []
    for p in bounds:
        rp = vsub(F, p, a.c)
        if not any(
            same_radius_point(F, rp, vsub(F, q, a.c)) for q in ordered
        ):
            ordered.append(p)
    if len(ordered) == 1:
        p = ordered[0]
        if on_closed_arc(F, arc_a, p) and on_closed_arc(F, arc_b, p):
            return [Contact(p, "tangent", pos_a=piece_position(F, a, p),
                            pos_b=piece_position(F, b, p))]
        return []
    cut = vsub(F, s_a, a.c)
    ordered.sort(key=lambda p: _Rank(F, cut, vsub(F, p, a.c)))

    m = len(ordered)
    inside: List[bool] = []
    for idx in range(m):
        p1, p2 = ordered[idx], ordered[(idx + 1) % m]
        mid = _bisector_point(F, a.circle, p1, p2)
        inside.append(
            on_closed_arc(F, arc_a, mid) and on_closed_arc(F, arc_b, mid)
        )

    out: List[Contact] = []
    idx = 0
    spans_seen = 0
    while idx < m:
        if not inside[idx]:
            idx += 1
            continue
        start_idx = idx
        while inside[idx % m]:
            idx += 1
            spans_seen += 1
            if idx - start_idx == m:
                break
        end_idx = idx % m
        p1, p2 = ordered[start_idx], ordered[end_idx]
        out.append(Contact(None, "overlap", interval=(p1, p2)))
        if spans_seen >= m:
            break
    if out:
        return out
    # No shared interior span: an isolated shared boundary point is a
    # tangential endpoint contact.
    for p in ordered:
        if on_closed_arc(F, arc_a, p) and on_closed_arc(F, arc_b, p):
            return [Contact(p, "tangent", pos_a=piece_position(F, a, p),
                            pos_b=piece_position(F, b, p))]
    return []


class _Rank:
    """Total CCW order rank of a radius vector from a fixed cut."""

    def __init__(self, F: Field, cut: Vec, r: Vec):
        self.F = F
        self.cut = cut
        self.r = r
        cr = F.sign(vcross(F, cut, r))
        dt = F.sign(vdot(F, cut, r))
        self.zero = same_radius_point(F, cut, r)
        self.half = 0 if (cr > 0 or (cr == 0 and dt > 0)) else 1

    def __lt__(self, other: "_Rank") -> bool:
        if self.zero:
            return False
        if other.zero:
            return True
        if self.half != other.half:
            return self.half < other.half
        return self.F.sign(vcross(self.F, self.r, other.r)) > 0


def _bisector_point(F: Field, circle: Circle, p1: Vec, p2: Vec) -> Vec:
    """A point on the circle strictly on the short CCW arc p1 -> p2.

    The normalized sum of the two radius vectors is the internal angle
    bisector and lands exactly between them (handles antipodal points).
    """
    r1, r2 = vsub(F, p1, circle.c), vsub(F, p2, circle.c)
    s = Vec(F.add(r1.x, r2.x), F.add(r1.y, r2.y))
    l2 = F.add(F.sq(s.x), F.sq(s.y))
    if F.is_zero(l2):
        s = Vec(F.neg(r1.y), r1.x)
        l2 = F.add(F.sq(s.x), F.sq(s.y))
    L = F.square_root(l2)
    return Vec(
        F.add(circle.c.x, F.mul(circle.r, F.div(s.x, L))),
        F.add(circle.c.y, F.mul(circle.r, F.div(s.y, L))),
    )


def piece_contacts(F: Field, a, b) -> List[Contact]:
    if isinstance(a, Seg) and isinstance(b, Seg):
        return seg_seg_contacts(F, a, b)
    if isinstance(a, Seg):
        return seg_arc_contacts(F, a, b)
    if isinstance(b, Seg):
        out = seg_arc_contacts(F, b, a)
        for c in out:
            c.pos_a, c.pos_b = c.pos_b, c.pos_a
        return out
    return arc_arc_contacts(F, a, b)


# ----------------------------------------------------------------- helpers
def min2(F: Field, x: Element, y: Element) -> Element:
    return x if F.le(x, y) else y


def max2(F: Field, x: Element, y: Element) -> Element:
    return x if F.ge(x, y) else y
