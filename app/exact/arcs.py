"""Circular arcs and exact angular predicates.

No trigonometry and no angle tolerances are ever used.  An arc is a circle
plus two boundary radius vectors and a sweep sense; whether an algebraic
point lies on an arc, and how two points are ordered along it, is decided
purely from signs of cross products -- exact sign decisions in the field.
"""

from __future__ import annotations

from dataclasses import dataclass

from .algebraic import Element, Field
from .geometry import Circle, Vec, vcross, vdot, vsub


CCW = 1
CW = -1


@dataclass
class Arc:
    circle: Circle
    start: Vec  # radius vector s = start_point - c
    end: Vec  # radius vector e = end_point - c
    sense: int  # CCW (+1) or CW (-1)
    full: bool = False  # full 2*pi sweep (start coincides with end)

    @property
    def c(self) -> Vec:
        return self.circle.c

    @property
    def r(self) -> Element:
        return self.circle.r


def radius_to(F: Field, arc: Arc, p: Vec) -> Vec:
    return vsub(F, p, arc.c)


def same_radius_point(F: Field, a: Vec, b: Vec) -> bool:
    """Two radius vectors on the same circle point at the same point."""
    return F.is_zero(vcross(F, a, b)) and F.sign(vdot(F, a, b)) > 0


def _on_ccw(F: Field, s: Vec, e: Vec, q: Vec) -> bool:
    """q on the closed CCW arc s -> e (sweep in (0, 2*pi), s != -e cases)."""
    se = F.sign(vcross(F, s, e))
    if se > 0:
        # Sweep in (0, pi): wedge between the two radius vectors.
        return F.sign(vcross(F, s, q)) >= 0 and F.sign(vcross(F, q, e)) >= 0
    if se < 0:
        # Sweep in (pi, 2pi): complement of the open short arc e -> s.
        in_gap = F.sign(vcross(F, e, q)) > 0 and F.sign(vcross(F, q, s)) > 0
        return not in_gap
    # collinear boundaries: sweep exactly pi (opposite radii) is the closed
    # CCW half-plane; sweep 0 / 2pi are handled by callers.
    if F.sign(vdot(F, s, e)) < 0:
        return F.sign(vcross(F, s, q)) >= 0
    # Degenerate s == e with non-full meaning: single point.
    return same_radius_point(F, s, q)


def on_closed_arc(F: Field, arc: Arc, p: Vec) -> bool:
    """Exact membership of ``p`` (known to be on the circle) in the arc."""
    if arc.full:
        return True
    q = radius_to(F, arc, p)
    if arc.sense == CCW:
        return _on_ccw(F, arc.start, arc.end, q)
    return _on_ccw(F, arc.end, arc.start, q)


def on_open_arc(F: Field, arc: Arc, p: Vec) -> bool:
    """Membership excluding the two endpoint points."""
    if arc.full:
        return True
    q = radius_to(F, arc, p)
    s, e = arc.start, arc.end
    if same_radius_point(F, q, s) or same_radius_point(F, q, e):
        return False
    if arc.sense == CCW:
        return _on_ccw(F, s, e, q)
    return _on_ccw(F, e, s, q)


def _halfplane_key(F: Field, s: Vec, q: Vec) -> int:
    """0 for CCW angle of q from s in [0, pi), 1 in [pi, 2pi)."""
    cr = F.sign(vcross(F, s, q))
    if cr > 0:
        return 0
    if cr < 0:
        return 1
    return 0 if F.sign(vdot(F, s, q)) > 0 else 1


def ccw_order_from(F: Field, s: Vec, a: Vec, b: Vec) -> int:
    """Order radius vectors a, b by CCW angle measured from s (a.s. distinct).

    Returns -1 if a precedes b, +1 if b precedes a, 0 if identical points.
    """
    if same_radius_point(F, a, b):
        return 0
    ha, hb = _halfplane_key(F, s, a), _halfplane_key(F, s, b)
    if ha != hb:
        return -1 if ha < hb else 1
    cr = F.sign(vcross(F, a, b))
    if cr != 0:
        return -1 if cr > 0 else 1
    # Opposite radii within the same key cannot happen; equal is covered.
    return 0


def traversal_order(F: Field, arc: Arc, p: Vec, q: Vec) -> int:
    """Compare two points of the arc in its traversal order.

    -1: p precedes q, +1: q precedes p, 0: same point.
    """
    a, b = radius_to(F, arc, p), radius_to(F, arc, q)
    if same_radius_point(F, a, b):
        return 0
    if arc.full:
        res = ccw_order_from(F, arc.start, a, b)
        return res if arc.sense == CCW else -res
    s, e = (arc.start, arc.end) if arc.sense == CCW else (arc.end, arc.start)
    # Both lie on the CCW arc s -> e; a precedes b iff b is reachable from a
    # without wrapping past e, i.e. b lies on the CCW arc a -> e.
    if _on_ccw(F, a, e, b):
        return -1 if arc.sense == CCW else 1
    return 1 if arc.sense == CCW else -1
