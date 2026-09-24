"""Compensated contour primitives.

A *piece* is one oriented segment of the final closed tool path:

* :class:`Seg`  -- an offset line segment (possibly a trimmed offset line);
* :class:`ArcP` -- a circle arc (an offset arc or a vertex fillet).

Coordinates live in a single exact :class:`~app.exact.algebraic.Field`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


from ..exact.algebraic import Element, Field
from ..exact.geometry import Circle, Vec, vscale, vsub


@dataclass
class Seg:
    a: Vec
    b: Vec
    u: Vec  # unit direction a -> b
    kind: str = "line"  # always "line"
    source: Optional[int] = None

    @property
    def start(self) -> Vec:
        return self.a

    @property
    def end(self) -> Vec:
        return self.b


@dataclass
class ArcP:
    circle: Circle
    start: Vec  # absolute points
    end: Vec
    sense: int  # +1 CCW, -1 CW
    full: bool = False
    kind: str = "arc"  # 'arc' | 'fillet'
    source: Optional[int] = None
    join_vertex: Optional[int] = None

    @property
    def c(self) -> Vec:
        return self.circle.c

    @property
    def r(self) -> Element:
        return self.circle.r

    @property
    def start_v(self) -> Vec:
        return self.start

    @property
    def end_v(self) -> Vec:
        return self.end


Piece = "Seg | ArcP"


def unit_tangent_line(F: Field, p: Vec, q: Vec) -> Vec:
    d = vsub(F, q, p)
    length = F.square_root(F.add(F.sq(d.x), F.sq(d.y)))
    return vscale(F, d, F.inv(length))


def unit_tangent_arc_at(F: Field, c: Vec, p: Vec, sense: int) -> Vec:
    """Unit traversal tangent of a CCW/CW arc at point p."""
    rv = vsub(F, p, c)
    length = F.square_root(F.add(F.sq(rv.x), F.sq(rv.y)))
    # CCW: (-y, x)/r ; CW: (y, -x)/r
    if sense > 0:
        t = Vec(F.neg(rv.y), rv.x)
    else:
        t = Vec(rv.y, F.neg(rv.x))
    return vscale(F, t, F.inv(length))


# ----------------------------------------------------------------- bbox
@dataclass
class Box:
    xmin: Element
    xmax: Element
    ymin: Element
    ymax: Element


def as_exact_arc(F: Field, arc: ArcP):
    from ..exact.arcs import Arc as EArc

    return EArc(
        arc.circle,
        vsub(F, arc.start, arc.c),
        vsub(F, arc.end, arc.c),
        arc.sense,
        arc.full,
    )


def piece_box(F: Field, piece) -> Box:
    """Exact bounding box of a piece (arc extrema tested exactly on-arc)."""
    from ..exact.arcs import on_closed_arc

    if isinstance(piece, Seg):
        xs = [piece.a.x, piece.b.x]
        ys = [piece.a.y, piece.b.y]
        return Box(min(xs, key=lambda e: _key(F, e)), max(xs, key=lambda e: _key(F, e)),
                   min(ys, key=lambda e: _key(F, e)), max(ys, key=lambda e: _key(F, e)))
    arc = piece
    pts = [arc.start, arc.end]
    c, r = arc.c, arc.r
    axis_pts = [
        Vec(F.add(c.x, r), c.y),
        Vec(F.sub(c.x, r), c.y),
        Vec(c.x, F.add(c.y, r)),
        Vec(c.x, F.sub(c.y, r)),
    ]
    ea = as_exact_arc(F, arc)
    for p in axis_pts:
        if on_closed_arc(F, ea, p):
            pts.append(p)
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return Box(
        min(xs, key=lambda e: _key(F, e)),
        max(xs, key=lambda e: _key(F, e)),
        min(ys, key=lambda e: _key(F, e)),
        max(ys, key=lambda e: _key(F, e)),
    )


class _Key:
    def __init__(self, F: Field, x: Element):
        self.F = F
        self.x = x

    def __lt__(self, other):
        return self.F.lt(self.x, other.x)

    def __eq__(self, other):
        return self.F.eq(self.x, other.x)


def _key(F: Field, x: Element) -> _Key:
    return _Key(F, x)


def boxes_may_overlap(F: Field, b1: Box, b2: Box) -> bool:
    return not (
        F.lt(b1.xmax, b2.xmin)
        or F.lt(b2.xmax, b1.xmin)
        or F.lt(b1.ymax, b2.ymin)
        or F.lt(b2.ymax, b1.ymin)
    )
