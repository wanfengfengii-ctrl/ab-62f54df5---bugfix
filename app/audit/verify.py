"""Global verification of the compensated chain.

Exact checks, in order of tool-path position:

1. chain continuity (piece end == next piece start) and piece regularity;
2. every pair of pieces: adjacent pairs may meet only at their shared joint;
   any other contact (crossing, tangency, overlap interval) is a defect;
3. orientation -- the turning number of the unit tangent is accumulated
   exactly.  Every tangent rotation is a unit complex number
   ``(dot, cross)/norm`` living in the field tower; crossing the branch cut
   is decided by the sign of one cross product.  Arc sweeps beyond pi are
   counted via the exact sign of the endpoint radius-vector cross product.
   No angle value, epsilon guess or sampling is used.

Signed area is returned as an exact recomputable structure (linear cross
terms plus sector terms carrying exact ``r**2``, ``dot`` and ``cross``)
together with a display-only decimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Set, Tuple

from ..exact.algebraic import Element, Field
from ..exact.geometry import (
    Vec,
    vcross,
    vdot,
    vsub,
)
from .intersections import (
    Contact,
    piece_contacts,
    piece_position,
    same_point,
)
from .primitives import Seg, unit_tangent_arc_at


@dataclass
class Defect:
    code: str
    message: str
    position: Dict[str, Any]
    sources: List[int]
    detail: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------- continuity
def check_continuity(F: Field, chain: List) -> Optional[Defect]:
    m = len(chain)
    for idx in range(m):
        a, b = chain[idx], chain[(idx + 1) % m]
        if not same_point(F, a.end, b.start):
            return Defect(
                "chain_discontinuity",
                f"piece {idx} does not connect exactly to piece "
                f"{(idx + 1) % m}",
                {"piece_index": idx, "at": "end"},
                _sources_of(a, b),
                {"end": a.end, "next_start": b.start},
            )
    return None


def check_regularity(F: Field, chain: List) -> Optional[Defect]:
    for idx, p in enumerate(chain):
        if isinstance(p, Seg):
            if same_point(F, p.a, p.b):
                return Defect(
                    "zero_length_segment",
                    f"piece {idx} collapsed to a single point",
                    {"piece_index": idx},
                    [p.source] if p.source is not None else [],
                )
            one = F.add(F.sq(p.u.x), F.sq(p.u.y))
            if not F.eq(one, F.rational(1)):
                return Defect(
                    "non_unit_direction",
                    f"piece {idx} direction is not exactly unit",
                    {"piece_index": idx},
                    [p.source] if p.source is not None else [],
                )
        else:
            if not (F.sign(p.r) > 0):
                return Defect(
                    "radius_collapse",
                    f"piece {idx} has non-positive radius",
                    {"piece_index": idx},
                    [p.source] if p.source is not None else [],
                )
            if not p.full and same_point(F, p.start, p.end):
                return Defect(
                    "ambiguous_closed_arc",
                    f"piece {idx} is a zero-sweep arc",
                    {"piece_index": idx},
                    [p.source] if p.source is not None else [],
                )
            for q in (p.start, p.end):
                d = vsub(F, q, p.c)
                if not F.eq(F.add(F.sq(d.x), F.sq(d.y)), F.sq(p.r)):
                    return Defect(
                        "endpoint_off_circle",
                        f"piece {idx} endpoint is not exactly on its circle",
                        {"piece_index": idx},
                        [p.source] if p.source is not None else [],
                    )
    return None


def _sources_of(a, b) -> List[int]:
    out: List[int] = []
    for p in (a, b):
        s = getattr(p, "source", None)
        if s is not None and s not in out:
            out.append(s)
    return out


# ------------------------------------------------------------- contact scan
def _scan_pairs(F: Field, chain: List):
    from .primitives import boxes_may_overlap, piece_box
    from .spatial import ExactGrid

    m = len(chain)
    boxes = [piece_box(F, p) for p in chain]
    grid = ExactGrid(F, boxes)
    pairs: Set[Tuple[int, int]] = set()
    # Adjacent pairs are always tested (stored with the smaller index
    # first, including the wrapping pair (0, m-1)); every other candidate
    # must come from the exact spatial grid.
    for i in range(m):
        a, b = i, (i + 1) % m
        if a != b:  # a single full-turn piece has no self-pair
            pairs.add((a, b) if a < b else (b, a))
    for i, j in grid.candidate_pairs():
        if i < j:
            pairs.add((i, j))
    for i, j in sorted(pairs):
        adjacent = (j == i + 1) or (i == 0 and j == m - 1)
        # Cheap exact bounding-box re-check for grid candidates.
        if not adjacent and not boxes_may_overlap(F, boxes[i], boxes[j]):
            continue
        yield i, j, adjacent, piece_contacts(F, chain[i], chain[j])


def contact_defects(F: Field, chain: List) -> List[Defect]:
    """All simplicity violations, ordered by tool-path position."""
    found: List[Defect] = []
    m = len(chain)
    for i, j, adjacent, contacts in _scan_pairs(F, chain):
        if adjacent:
            # Allowed joints are exactly the endpoints the two pieces share
            # by chain adjacency.  With m == 2 the same pair is both the
            # forward pair (0,1) and the wrapping pair, so both endpoints
            # (chain[0].end == chain[1].start and chain[1].end == chain[0].
            # start) are legitimate joints.
            allowed = [chain[i].end if j == i + 1 else chain[j].end]
            if m == 2:
                allowed.append(chain[j].end if j == i + 1 else chain[i].end)
        else:
            allowed = []
        for c in contacts:
            if (
                adjacent
                and c.kind != "overlap"
                and c.point is not None
                and any(same_point(F, c.point, q) for q in allowed)
            ):
                continue
            found.append(_contact_to_defect(F, chain, i, j, adjacent, c))
        if adjacent and not contacts:
            found.append(
                Defect(
                    "chain_discontinuity",
                    f"adjacent pieces {i} and {j} share no exact point",
                    {"piece_index": i, "at": "end"},
                    _sources_of(chain[i], chain[j]),
                )
            )
    found.sort(key=lambda d: d.position["order_key"])
    return found


def _contact_to_defect(
    F: Field, chain, i: int, j: int, adjacent: bool, c: Contact
) -> Defect:
    srcs = _sources_of(chain[i], chain[j])
    if c.kind == "overlap":
        p = c.interval[0]
        code = "adjacent_overlap" if adjacent else "non_adjacent_overlap"
        message = (
            f"pieces {i} and {j} coincide over an interval instead of "
            "forming a simple boundary"
        )
        detail = {"interval_start": c.interval[0], "interval_end": c.interval[1]}
    else:
        p = c.point
        if adjacent:
            code = "adjacent_extra_contact"
            message = (
                f"adjacent pieces {i} and {j} touch away from their shared "
                "joint"
            )
        elif c.kind == "tangent":
            code = "non_adjacent_contact"
            message = (
                f"non-adjacent pieces {i} and {j} are exactly tangent at an "
                "isolated point (the curve pinches; it is not simple)"
            )
        else:
            code = "self_intersection"
            message = f"non-adjacent pieces {i} and {j} intersect transversely"
        detail = {"point": p, "contact": c.kind}
    ranks = []
    for piece_idx in (i, j):
        rank = piece_position(F, chain[piece_idx], p)
        ranks.append((piece_idx * 2 + rank, piece_idx, rank))
    _, piece_idx, rank = min(ranks, key=lambda t: t[0])
    return Defect(
        code,
        message,
        {
            "piece_index": piece_idx,
            "intra": ("start", "interior", "end")[rank],
            "order_key": piece_idx * 2 + rank,
            "pieces": [i, j],
            "point": p,
        },
        srcs,
        detail,
    )


# ----------------------------------------------- exact tangent turning number
def _piece_end_tangents(F: Field, p) -> Tuple[Vec, Vec, Optional[dict]]:
    """Unit tangents at piece start/end and the internal sweep description.

    The sweep is returned as (turns, ca, sa): a unit complex number
    (ca + i sa) for the residual rotation in (-pi, pi], plus an integer
    number of full turns carried by long arcs.
    """
    zero = F.rational(0)
    one = F.rational(1)
    if isinstance(p, Seg):
        return p.u, p.u, {"turns": 0, "ca": one, "sa": zero}
    ts = unit_tangent_arc_at(F, p.c, p.start, p.sense)
    te = unit_tangent_arc_at(F, p.c, p.end, p.sense)
    if p.full:
        return ts, ts, {"turns": p.sense, "ca": one, "sa": zero}
    # Signed sweep sigma of the radius vectors: cos sigma = dot/r^2,
    # sin sigma = cross/r^2, independent of sense; sense decides how the
    # principal (-pi, pi) angle is lifted to the actual signed sweep.
    ra = vsub(F, p.start, p.c)
    rb = vsub(F, p.end, p.c)
    cr = vcross(F, ra, rb)
    dt = vdot(F, ra, rb)
    inv_r2 = F.inv(F.sq(p.r))
    ca = F.mul(dt, inv_r2)
    sa = F.mul(cr, inv_r2)
    sc = F.sign(cr)
    turns = 0
    if sc == 0 and F.sign(dt) < 0:
        # Diametrically opposite endpoints: sweep is exactly +/- pi.  The
        # rotation primitive counts +pi; a CW semicircle is -pi and is
        # obtained by subtracting one full turn.
        if p.sense < 0:
            turns = -1
    elif p.sense > 0 and sc < 0:
        turns = 1  # sigma = principal + 2*pi
    elif p.sense < 0 and sc > 0:
        turns = -1  # sigma = principal - 2*pi
    return ts, te, {"turns": turns, "ca": ca, "sa": sa}


def _apply_rotation(
    F: Field, z: Tuple[Element, Element], ca: Element, sa: Element
) -> Tuple[Tuple[Element, Element], int]:
    """Rotate accumulated angle z by a = atan2(sa, ca) in (-pi, pi].

    ``z`` carries an angle theta in [0, 2*pi).  Returns (new z, integer wrap
    added to the turning count).  Branch-cut crossings are decided by exact
    cross-product signs against the short boundary arc on the unit circle.
    """
    zx, zy = z
    wx = F.sub(F.mul(zx, ca), F.mul(zy, sa))
    wy = F.add(F.mul(zx, sa), F.mul(zy, ca))
    s_sa = F.sign(sa)
    wrap = 0
    if s_sa > 0:
        # Wrap +1 iff theta in [2*pi - a, 2*pi): z on the short CCW arc
        # (ca, -sa) -> (1, 0), excluding theta = 0 (which is the (1,0)
        # endpoint lying in the upper-boundary tie).
        q = F.add(F.mul(ca, zy), F.mul(sa, zx))
        if F.sign(q) >= 0 and F.sign(zy) < 0:
            wrap = 1
    elif s_sa < 0:
        # Wrap -1 iff theta in [0, -a): inside the short CCW arc
        # (1, 0) -> (ca, -sa), excluding its terminal endpoint.
        if F.sign(zy) >= 0 and F.sign(
            F.sub(F.mul(F.neg(sa), zx), F.mul(ca, zy))
        ) > 0:
            wrap = -1
    else:
        # a == pi (ca == -1, sa == 0): theta + pi >= 2*pi iff theta >= pi.
        if F.sign(ca) < 0:
            sy = F.sign(zy)
            if sy < 0 or (sy == 0 and F.sign(zx) < 0):
                wrap = 1
    return (wx, wy), wrap


def turning_number(F: Field, chain: List) -> Tuple[int, List[dict]]:
    """Exact total turning number (2*pi*k for any regular closed curve)."""
    m = len(chain)
    tangents_s: List[Vec] = []
    tangents_e: List[Vec] = []
    sweeps: List[dict] = []
    for p in chain:
        ts, te, sw = _piece_end_tangents(F, p)
        tangents_s.append(ts)
        tangents_e.append(te)
        sweeps.append(sw)

    turns = 0
    z = (F.rational(1), F.rational(0))
    steps: List[dict] = []
    for idx in range(m):
        # Rotation internal to piece idx.
        sw = sweeps[idx]
        turns += sw["turns"]
        z, w = _apply_rotation(F, z, sw["ca"], sw["sa"])
        turns += w
        steps.append({"at": f"piece:{idx}", "wrap": sw["turns"] + w})
        # Exterior rotation from end tangent of idx to start tangent of idx+1.
        ta, tb = tangents_e[idx], tangents_s[(idx + 1) % m]
        cr = vcross(F, ta, tb)
        dt = vdot(F, ta, tb)
        if F.is_zero(cr) and F.sign(dt) > 0:
            continue
        z, w = _apply_rotation(F, z, dt, cr)
        turns += w
        steps.append({"at": f"vertex:{(idx + 1) % m}", "wrap": w})
    # Closure: z must be exactly back to (1, 0).
    if not (F.is_zero(F.sub(z[0], F.rational(1))) and F.is_zero(z[1])):
        return turns, steps + [{"closure_error": True}]
    return turns, steps


# --------------------------------------------------------------- signed area
def signed_area_parts(F: Field, chain: List):
    """Exact symbolic 1/2 * integral (x dy - y dx) contributions."""
    line_terms: List[Element] = []
    arc_terms: List[Dict[str, Any]] = []
    for p in chain:
        if isinstance(p, Seg):
            line_terms.append(
                F.scale(
                    F.sub(F.mul(p.a.x, p.b.y), F.mul(p.b.x, p.a.y)),
                    Fraction(1, 2),
                )
            )
        else:
            c, a0, b0 = p.c, p.start, p.end
            # Arc contribution of 1/2 integral (x dy - y dx):
            # 1/2 * [cross(c, B - A) + r**2 * sigma].  There is deliberately
            # no chord term here; the line pieces already contribute their
            # own chords.
            line_terms.append(
                F.scale(
                    F.add(
                        F.mul(c.x, F.sub(b0.y, a0.y)),
                        F.mul(F.neg(c.y), F.sub(b0.x, a0.x)),
                    ),
                    Fraction(1, 2),
                )
            )
            ra, rb = vsub(F, a0, c), vsub(F, b0, c)
            arc_terms.append(
                {
                    "radius_squared": F.sq(p.r),
                    "cross": vcross(F, ra, rb),
                    "dot": vdot(F, ra, rb),
                    "sense": p.sense,
                    "full": p.full,
                }
            )
    return line_terms, arc_terms
