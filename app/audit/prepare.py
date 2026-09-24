"""Turn the request into exact rational source entities.

Validation performed here, all exactly on the original decimal values:

* every entity's end coincides with the next entity's start (chain closure);
* lines have distinct endpoints; arcs have a center, a sense, equal-radius
  endpoints and a non-zero radius;
* compensation side is resolved per entity.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Tuple

from ..exact.number import parse_decimal
from .build import SourceEnt, BuildError, LEFT, RIGHT


def _point(values) -> Tuple[Fraction, Fraction]:
    return parse_decimal(values[0]), parse_decimal(values[1])


def prepare_entities(payload) -> Tuple[List[SourceEnt], Fraction, int]:
    n = len(payload.contour)
    tool = parse_decimal(payload.tool_radius)
    if tool <= 0:
        raise BuildError(
            "invalid_tool_radius",
            f"tool radius must be strictly positive, got {payload.tool_radius!r}",
            list(range(n)),
        )
    global_tau = LEFT if payload.side == "left" else RIGHT
    raws = payload.contour
    ents: List[SourceEnt] = []
    points: List[Tuple[Fraction, Fraction]] = []

    for i, r in enumerate(raws):
        start = _point(r.start)
        end = _point(r.end)
        label = r.id or str(i)
        tau = (
            global_tau
            if r.side is None
            else (LEFT if r.side == "left" else RIGHT)
        )
        if r.type == "line":
            if start == end:
                raise BuildError(
                    "degenerate_line",
                    f"entity {i} ({label}): zero-length line, identical endpoints",
                    [i],
                    point=start,
                )
            center = None
            sense = 0
            full = False
            radius2 = Fraction(0)
        else:
            if r.center is None:
                raise BuildError(
                    "arc_without_center",
                    f"entity {i} ({label}): arc requires an exact center",
                    [i],
                )
            if r.sense is None:
                raise BuildError(
                    "arc_without_sense",
                    f"entity {i} ({label}): arc requires 'sense' of 'ccw' or 'cw'",
                    [i],
                )
            center = _point(r.center)
            sense = 1 if r.sense == "ccw" else -1
            rs = (start[0] - center[0]) ** 2 + (start[1] - center[1]) ** 2
            re_ = (end[0] - center[0]) ** 2 + (end[1] - center[1]) ** 2
            if rs != re_:
                raise BuildError(
                    "arc_endpoint_radius_mismatch",
                    f"entity {i} ({label}): squared radii to center differ "
                    f"exactly ({rs} != {re_})",
                    [i],
                )
            if rs == 0:
                raise BuildError(
                    "radius_collapse",
                    f"entity {i} ({label}): arc radius is zero",
                    [i],
                )
            radius2 = rs
            full = bool(r.full)
            if start == end and not full:
                raise BuildError(
                    "ambiguous_closed_arc",
                    f"entity {i} ({label}): arc starts and ends at the same "
                    "point; set \"full\": true for a complete turn",
                    [i],
                )
            if full and start != end:
                raise BuildError(
                    "full_arc_endpoint_mismatch",
                    f"entity {i} ({label}): full arc must start and end at "
                    "the same point",
                    [i],
                )
        ents.append(
            SourceEnt(
                index=i,
                kind=r.type,
                start=start,
                end=end,
                center=center,
                sense=sense,
                full=full,
                tau=tau,
                label=label,
                radius2=radius2,
            )
        )
        points.append((start, end))

    # Chain connectivity in machining order (exact rational equality).
    for i in range(n):
        end_here = ents[i].end
        start_next = ents[(i + 1) % n].start
        if end_here != start_next:
            raise BuildError(
                "chain_not_closed",
                f"entity {i} ends at {end_here} but entity "
                f"{(i + 1) % n} starts at {start_next}",
                [i, (i + 1) % n],
                join_index=(i + 1) % n,
                end_point=end_here,
                start_point=start_next,
            )
    return ents, tool, global_tau
