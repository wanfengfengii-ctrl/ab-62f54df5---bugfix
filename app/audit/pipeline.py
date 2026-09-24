"""End-to-end audit pipeline orchestrating the exact engine."""

from __future__ import annotations

from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional

from ..exact.algebraic import Element, Field
from ..exact.encode import encode_number, encode_point
from .build import BuildError, OffsetBuilder
from .prepare import prepare_entities
from .primitives import Seg
from .verify import (
    Defect,
    check_continuity,
    check_regularity,
    contact_defects,
    signed_area_parts,
    turning_number,
)


class AuditFailure(Exception):
    def __init__(self, code: str, http: int, payload: Dict[str, Any]):
        super().__init__(payload.get("error", {}).get("message", code))
        self.code = code
        self.http = http
        self.payload = payload


# Build-stage conditions that mean the compensation is geometrically
# unsafe (as opposed to malformed input).
_UNSAFE_BUILD_CODES = {
    "radius_collapse",
    "radius_inversion",
    "trim_order_reversal",
    "traversal_reversal",
    "coincident_supports",
    "join_intersection_missing",
    "ambiguous_join_intersection",
    "gouging_corner",
    "compensation_side_switch",
    "tangent_join_mismatch",
}


def _build_unsafe_payload(F: Field, e: BuildError, places: int, ents) -> Dict[str, Any]:
    detail = {}
    point = None
    for k, v in e.extra.items():
        if k in ("start_trim", "end_trim", "point", "point_a", "point_b",
                 "tangent_point_a", "tangent_point_b"):
            detail[k] = encode_point(F, v, places)
        else:
            detail[k] = _stringify_extra(F, v)
    if "start_trim" in detail:
        point = e.extra.get("start_trim")
    elif "point" in e.extra:
        point = e.extra["point"]
    join_index = e.extra.get("join_index")
    position = {
        "piece_index": e.sources[0] if e.sources else 0,
        "intra": "interior",
        "order_key": (e.sources[0] * 2 + 1) if e.sources else 0,
        **({"join_vertex": join_index} if join_index is not None else {}),
    }
    return {
        "status": "unsafe",
        "error": {
            "code": e.code,
            "message": e.message,
            "first_defect": {
                "tool_path_position": position,
                "source_entities": _source_labels(ents, e.sources),
                "exact_point": encode_point(F, point, places) if point is not None else None,
                "detail": detail,
            },
        },
    }


def run_audit(payload) -> Dict[str, Any]:
    F = Field()
    places = payload.decimal_places
    try:
        ents, tool, global_tau = prepare_entities(payload)
    except BuildError as e:
        raise AuditFailure(
            e.code,
            422,
            _build_error_payload(payload, e),
        )

    builder = OffsetBuilder(F, ents, tool)
    try:
        chain, joins = builder.build()
    except BuildError as e:
        if e.code in _UNSAFE_BUILD_CODES:
            raise AuditFailure(
                e.code, 409, _build_unsafe_payload(F, e, places, ents)
            )
        raise AuditFailure(
            e.code,
            422,
            _build_error_payload(payload, e, F),
        )

    cont = check_continuity(F, chain)
    if cont is not None:
        raise AuditFailure(
            cont.code, 409, _defect_payload(payload, F, cont, places, ents, chain)
        )
    reg = check_regularity(F, chain)
    if reg is not None:
        raise AuditFailure(
            reg.code, 409, _defect_payload(payload, F, reg, places, ents, chain)
        )

    defects = contact_defects(F, chain)
    if defects:
        d = defects[0]
        raise AuditFailure(
            d.code, 409, _defect_payload(payload, F, d, places, ents, chain, defects)
        )

    tn, steps = turning_number(F, chain)
    if tn == 0:
        d = Defect(
            "winding_change",
            "the compensated contour has total turning number 0: it is not "
            "a single consistently oriented simple closed curve",
            {"piece_index": 0, "intra": "start", "order_key": 0},
            list(range(len(ents))),
            {"turning_number": 0},
        )
        raise AuditFailure(
            d.code, 409, _defect_payload(payload, F, d, places, ents, chain)
        )
    if abs(tn) != 1:
        d = Defect(
            "winding_change",
            f"the compensated contour winds {tn} times; the programmed "
            "contour orientation changed under compensation",
            {"piece_index": 0, "intra": "start", "order_key": 0},
            list(range(len(ents))),
            {"turning_number": tn},
        )
        raise AuditFailure(
            d.code, 409, _defect_payload(payload, F, d, places, ents, chain)
        )

    # Programmed contour turning number for the "direction unchanged" rule.
    src_turns = _source_turning_number(F, ents)
    if src_turns != 0 and src_turns != tn:
        d = Defect(
            "orientation_reversed",
            "the compensated orientation is opposite to the programmed "
            "traversal direction",
            {"piece_index": 0, "intra": "start", "order_key": 0},
            list(range(len(ents))),
            {
                "programmed_turning_number": src_turns,
                "compensated_turning_number": tn,
            },
        )
        raise AuditFailure(
            d.code, 409, _defect_payload(payload, F, d, places, ents, chain)
        )

    line_terms, arc_terms = signed_area_parts(F, chain)
    return _success_payload(payload, F, ents, chain, joins, tn, line_terms,
                            arc_terms, places)


# --------------------------------------------------------- source orientation
def _source_turning_number(F: Field, ents) -> int:
    """Turning number of the programmed rational contour (shoelace sign).

    Exact rational 2x determinant sum works for arcs as well because arcs do
    not change the winding of their chord polygon when the contour is simple
    and the arc sweep is < 2*pi; a robust computation reconstructs tangent
    turning using the same exact primitive.
    """
    # Reuse the exact turning engine on rational pieces is not available for
    # sources; compute tangent turns algebraically with rational unit vectors
    # embedded in F (roots create no sign ambiguity: all decisions are cross
    # signs of rational vectors).
    n = len(ents)
    from ..exact.geometry import Vec as EV
    from .primitives import unit_tangent_arc_at, unit_tangent_line

    tang_s: List[EV] = []
    tang_e: List[EV] = []
    sweeps: List[tuple] = []
    for e in ents:
        sp = EV(F.rational(e.start[0]), F.rational(e.start[1]))
        ep = EV(F.rational(e.end[0]), F.rational(e.end[1]))
        if e.kind == "line":
            u = unit_tangent_line(F, sp, ep)
            tang_s.append(u)
            tang_e.append(u)
            sweeps.append(None)
        else:
            cp = EV(F.rational(e.center[0]), F.rational(e.center[1]))
            ts = unit_tangent_arc_at(F, cp, sp, e.sense)
            te = unit_tangent_arc_at(F, cp, ep, e.sense)
            tang_s.append(ts)
            tang_e.append(te)
            ra = (e.start[0] - e.center[0], e.start[1] - e.center[1])
            rb = (e.end[0] - e.center[0], e.end[1] - e.center[1])
            cr = ra[0] * rb[1] - ra[1] * rb[0]
            sweeps.append((cr, e.sense, e.full))

    from .verify import _apply_rotation

    turns = 0
    z = (F.rational(1), F.rational(0))
    for i, e in enumerate(ents):
        if e.kind == "arc":
            cr, sense, full = sweeps[i]
            if full:
                turns += sense
            else:
                ra = EV(F.rational(e.start[0] - e.center[0]),
                        F.rational(e.start[1] - e.center[1]))
                rb = EV(F.rational(e.end[0] - e.center[0]),
                        F.rational(e.end[1] - e.center[1]))
                rho = F.square_root(F.rational(e.radius2))
                inv = F.inv(rho)
                ux, uy = F.mul(ra.x, inv), F.mul(ra.y, inv)
                vx, vy = F.mul(rb.x, inv), F.mul(rb.y, inv)
                ca = F.add(F.mul(ux, vx), F.mul(uy, vy))
                sa = F.sub(F.mul(ux, vy), F.mul(uy, vx))
                z, w = _apply_rotation(F, z, ca, sa)
                turns += w
                if cr == 0:
                    # Antipodal semicircle: the primitive contributes +pi;
                    # CW traversal sweeps -pi.
                    if sense < 0:
                        turns -= 1
                elif sense > 0 and cr < 0:
                    turns += 1
                elif sense < 0 and cr > 0:
                    turns -= 1
        ta, tb = tang_e[i], tang_s[(i + 1) % n]
        z, w = _apply_rotation(
            F, z,
            F.add(F.mul(ta.x, tb.x), F.mul(ta.y, tb.y)),
            F.sub(F.mul(ta.x, tb.y), F.mul(ta.y, tb.x)),
        )
        turns += w
    return turns


# -------------------------------------------------------------- serialization
def _success_payload(payload, F, ents, chain, joins, tn, line_terms, arc_terms,
                     places) -> Dict[str, Any]:
    pieces_out = []
    for idx, p in enumerate(chain):
        if isinstance(p, Seg):
            pieces_out.append(
                {
                    "index": idx,
                    "type": "line",
                    "start": encode_point(F, p.a, places),
                    "end": encode_point(F, p.b, places),
                    "source_entity": ents[p.source].label,
                    "source_index": p.source,
                }
            )
        else:
            pieces_out.append(
                {
                    "index": idx,
                    "type": p.kind,
                    "center": encode_point(F, p.c, places),
                    "radius": encode_number(F, p.r, places),
                    "start": encode_point(F, p.start, places),
                    "end": encode_point(F, p.end, places),
                    "sense": "ccw" if p.sense > 0 else "cw",
                    "full": p.full,
                    **(
                        {"source_entity": ents[p.source].label,
                         "source_index": p.source}
                        if p.source is not None
                        else {"join_vertex": p.join_vertex}
                    ),
                }
            )
    area_exact, area_decimal = _area_value(F, line_terms, arc_terms, places)
    return {
        "status": "safe",
        "compensated_contour": {
            "orientation": "ccw" if tn > 0 else "cw",
            "turning_number": tn,
            "pieces": pieces_out,
            "signed_area": {"exact": area_exact, "decimal": area_decimal},
        },
        "normalization": {
            "pieces": len(chain),
            "coordinates_exact": True,
            "decimal_places": places,
        },
    }


def _area_value(F, line_terms, arc_terms, places):
    """Sum area pieces exactly where possible and render.

    Sector terms carry a signed sweep angle theta with cos=dot/r^2,
    sin=cross/r^2; the term is (1/2) r^2 theta with theta lifted by sense.
    Its decimal value is computed at high precision with atan2; the exact
    structure returned lets any client re-derive it.
    """
    from decimal import Decimal as D

    guard = places + 16
    getcontext().prec = max(guard, 50)
    total = D(0)
    exact_line = []
    for t in line_terms:
        exact_line.append({"rational_cross_half": _rational_struct(F, t)})
        total += _element_decimal(F, t, guard)

    exact_sectors = []
    for s in arc_terms:
        theta_dec, theta_exact, turns = _sector_angle(F, s, guard)
        r2_dec = _element_decimal(F, s["radius_squared"], guard)
        total += D(1) / D(2) * r2_dec * theta_dec
        exact_sectors.append(
            {
                "radius_squared": _number_ref(F, s["radius_squared"]),
                "sweep": theta_exact,
                "full_turns": turns,
                "sense": s["sense"],
            }
        )
    q = D(1).scaleb(-places)
    return (
        {"linear_terms": exact_line, "circular_sectors": exact_sectors},
        format(total.quantize(q), "f"),
    )


def _sector_angle(F, s, guard):
    """Exact sweep-angle structure plus its fixed-guard decimal."""
    r2 = s["radius_squared"]
    cr, dt = s["cross"], s["dot"]
    if s.get("full"):
        theta = s["sense"] * _two_pi()
        turns = s["sense"]
        exact = {
            "full_turn": True,
            "angle": "2*pi" if s["sense"] > 0 else "-2*pi",
        }
        return theta, exact, turns
    cr_dec = _element_decimal(F, cr, guard)
    dt_dec = _element_decimal(F, dt, guard)
    sc = F.sign(cr)
    if sc == 0 and F.sign(dt) < 0:
        # Antipodal endpoints: the sweep is exactly +/- pi.
        theta = s["sense"] * _pi()
        turns = 0
        exact = {"semicircle": True, "angle": "pi" if s["sense"] > 0 else "-pi"}
        return theta, exact, turns
    theta = _atan2_decimal(cr_dec, dt_dec)
    turns = 0
    if s["sense"] > 0 and sc < 0:
        theta += _two_pi()
        turns = 1
    elif s["sense"] < 0 and sc > 0:
        theta -= _two_pi()
        turns = -1
    exact = {
        "cos": _number_ref(F, F.div(dt, r2)),
        "sin": _number_ref(F, F.div(cr, r2)),
        "angle": "atan2(sin, cos) + 2*pi*full_turns",
    }
    return theta, exact, turns


_PI_TEXT = (
    "3.141592653589793238462643383279502884197169399375105820974944592307816406"
    "28620899862803482534211706798214808651328230664709384460955058223172535940"
)


def _pi() -> Decimal:
    # 100+ digits; display precision never exceeds decimal_places + guard.
    return Decimal(_PI_TEXT)


def _two_pi() -> Decimal:
    return 2 * _pi()


def _atan_decimal(x: Decimal) -> Decimal:
    """arctan(x) via argument reduction to |z| <= sqrt(2)-1 + series."""
    if x < 0:
        return -_atan_decimal(-x)
    pi4 = _pi() / 4
    if x > 1:
        return _pi() / 2 - _atan_decimal(Decimal(1) / x)
    t = Decimal(2).sqrt() - 1  # tan(pi/8)
    # atan(x) = pi/4 + atan((x - 1)/(x + 1)); reduces |argument| to <= t.
    if x > t:
        z = (x - 1) / (x + 1)
        return pi4 + _atan_small(z)
    return _atan_small(x)


def _atan_small(z: Decimal) -> Decimal:
    """arctan series, converging quickly for |z| <= tan(pi/8)."""
    z2 = z * z
    term = z
    total = z
    k = 1
    eps = Decimal(1).scaleb(-(getcontext().prec - 4))
    while abs(term) > eps:
        term *= z2
        k += 2
        delta = term / Decimal(k)
        if k % 4 == 1:
            total += delta
        else:
            total -= delta
    return total


def _atan2_decimal(y: Decimal, x: Decimal) -> Decimal:
    """High-precision atan2 in (-pi, pi] using only Decimal arithmetic."""
    zero = Decimal(0)
    if y == 0:
        return zero if x > 0 else _pi()
    if x == 0:
        return _pi() / 2 if y > 0 else -_pi() / 2
    a = _atan_decimal(abs(y) / abs(x))
    if x > 0:
        return a if y > 0 else -a
    return (_pi() - a) if y > 0 else -( _pi() - a)


def _element_decimal(F, x: Element, places: int) -> Decimal:
    from ..exact.encode import evaluate_decimal

    return evaluate_decimal(F, x, places)


def _rational_struct(F, x) -> Dict[str, Any]:
    if len(x.terms) == 1 and () in x.terms:
        a = x.terms[()]
        return {"numerator": a.numerator, "denominator": a.denominator}
    return _number_ref(F, x)


def _number_ref(F, x) -> Dict[str, Any]:
    from ..exact.encode import encode_exact

    return encode_exact(F, F.reduce(x))


# --------------------------------------------------------------- error bodies
def _build_error_payload(payload, e: BuildError, F: Optional[Field] = None):
    detail = {}
    for k, v in e.extra.items():
        detail[k] = _stringify_extra(F, v)
    return {
        "status": "rejected",
        "error": {
            "code": e.code,
            "message": e.message,
            "source_entities": e.sources,
            "detail": detail,
        },
    }


def _defect_payload(payload, F, d: Defect, places, ents, chain, all_defects=None):
    pos = dict(d.position)
    point = pos.pop("point", None)
    detail = {}
    for k, v in d.detail.items():
        detail[k] = _stringify_extra(F, v)
    body = {
        "status": "unsafe",
        "error": {
            "code": d.code,
            "message": d.message,
            "first_defect": {
                "tool_path_position": pos,
                "source_entities": _source_labels(ents, d.sources),
                "exact_point": encode_point(F, point, places) if point is not None else None,
                "detail": detail,
            },
        },
    }
    if all_defects is not None and len(all_defects) > 1:
        body["error"]["defect_count"] = len(all_defects)
    return body


def _source_labels(ents, indices):
    out = []
    for i in indices:
        if 0 <= i < len(ents):
            out.append({"index": i, "id": ents[i].label})
    return out


def _stringify_extra(F, value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_stringify_extra(F, v) for v in value]
    if isinstance(value, dict):
        return {k: _stringify_extra(F, v) for k, v in value.items()}
    # Exact Vec or Element.
    from ..exact.geometry import Vec

    if isinstance(value, Vec):
        return encode_point(F, value)
    if isinstance(value, Element):
        return encode_number(F, value)
    return repr(value)
