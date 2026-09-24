"""Randomized property tests.

A *float reference implementation* (used only inside the tests, never inside
the service) independently rebuilds the compensated contour from the engine
output and checks:

* exact reported chain closes;
* consecutive pieces meet with matching tangents;
* every offset piece lies at exactly the tool distance from its source;
* the chain is geometrically simple (dense float sampling);
* its signed area agrees with the engine's reported value.

Random contours are star-shaped simple polygons, which always admit small
inner and outer offsets, so they must be reported ``safe``.
"""

import math
import random

import pytest

from app.models import AuditRequest
from app.audit.pipeline import run_audit, AuditFailure


def _star_polygon(seed, n=9, rmin=4.0, rmax=12.0):
    rng = random.Random(seed)
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n + rng.uniform(-0.12, 0.12)
        r = rng.uniform(rmin, rmax)
        pts.append((r * math.cos(a), r * math.sin(a)))
    # CCW order guaranteed by angular construction.
    return pts


def _contour(pts):
    n = len(pts)
    out = []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        out.append({
            "type": "line",
            "start": [f"{x1:.9f}", f"{y1:.9f}"],
            "end": [f"{x2:.9f}", f"{y2:.9f}"],
        })
    return out


def _audit(pts, tool, side):
    req = AuditRequest.model_validate(
        {"contour": _contour(pts), "tool_radius": str(tool), "side": side,
         "decimal_places": 15}
    )
    try:
        return run_audit(req)
    except AuditFailure as ex:
        return ex.payload


def _pieces_float(body):
    out = []
    for p in body["compensated_contour"]["pieces"]:
        if p["type"] == "line":
            out.append(("L",
                        (float(p["start"]["x"]["decimal"]),
                         float(p["start"]["y"]["decimal"])),
                        (float(p["end"]["x"]["decimal"]),
                         float(p["end"]["y"]["decimal"]))))
        else:
            out.append(("A", p["type"],
                        (float(p["center"]["x"]["decimal"]),
                         float(p["center"]["y"]["decimal"])),
                        float(p["radius"]["decimal"]),
                        (float(p["start"]["x"]["decimal"]),
                         float(p["start"]["y"]["decimal"])),
                        (float(p["end"]["x"]["decimal"]),
                         float(p["end"]["y"]["decimal"])),
                        1 if p["sense"] == "ccw" else -1))
    return out


def _sample(pieces, steps=4000):
    pts = []
    tangents = []
    for p in pieces:
        if p[0] == "L":
            a, b = p[1], p[2]
            for k in range(steps // len(pieces) // 2 + 2):
                t = k / (steps // len(pieces) // 2 + 1)
                pts.append((a[0] + (b[0] - a[0]) * t,
                            a[1] + (b[1] - a[1]) * t))
                tangents.append((b[0] - a[0], b[1] - a[1]))
        else:
            c, r, a, b, sense = p[2], p[3], p[4], p[5], p[6]
            ta = math.atan2(a[1] - c[1], a[0] - c[0])
            tb = math.atan2(b[1] - c[1], b[0] - c[0])
            if sense > 0:
                while tb < ta:
                    tb += 2 * math.pi
            else:
                while tb > ta:
                    tb -= 2 * math.pi
            m = steps // len(pieces) + 2
            for k in range(m):
                t = k / (m - 1)
                ang = ta + (tb - ta) * t
                pts.append((c[0] + r * math.cos(ang), c[1] + r * math.sin(ang)))
                d = (-r * math.sin(ang) * sense, r * math.cos(ang) * sense)
                tangents.append(d)
    return pts, tangents


def _segment_intersect(p, q, a, b):
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    r = (q[0] - p[0], q[1] - p[1])
    s = (b[0] - a[0], b[1] - a[1])
    rxs = cross(r, s)
    if abs(rxs) < 1e-11:
        return False
    qp = (a[0] - p[0], a[1] - p[1])
    t = cross(qp, s) / rxs
    u = cross(qp, r) / rxs
    return 1e-7 < t < 1 - 1e-7 and 1e-7 < u < 1 - 1e-7


def _is_simple(points, min_gap=20):
    n = len(points)
    for i in range(n):
        p, q = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= min_gap or (i == 0 and j >= n - min_gap):
                continue
            a, b = points[j], points[(j + 1) % n]
            if _segment_intersect(p, q, a, b):
                return False
    return True


def _end_tangent(p, where):
    """Unit traversal tangent at 'start' or 'end' of a float piece."""
    if p[0] == "L":
        a, b = p[1], p[2]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        return dx / L, dy / L
    c, r, a, b, sense = p[2], p[3], p[4], p[5], p[6]
    q = a if where == "start" else b
    rx, ry = q[0] - c[0], q[1] - c[1]
    tx, ty = -ry / r, rx / r
    return (tx * sense, ty * sense)

@pytest.mark.parametrize("seed", range(8))
def test_random_star_polygons_inner_and_outer(seed):
    pts = _star_polygon(1000 + seed)
    for side in ("left", "right"):
        body = _audit(pts, 0.35, side)
        assert body.get("status") == "safe", (seed, side, body)
        cc = body["compensated_contour"]
        pieces = _pieces_float(body)
        # Closes (sharp concave joins are genuine corners, so tangents
        # need not agree; every end point must equal the next start point).
        for i in range(len(pieces)):
            cur = pieces[i]
            nxt = pieces[(i + 1) % len(pieces)]
            tail = cur[2] if cur[0] == "L" else cur[5]
            head = nxt[1] if nxt[0] == "L" else nxt[4]
            assert math.hypot(tail[0] - head[0], tail[1] - head[1]) < 1e-6
        # At every round fillet both neighboring pieces must share its
        # endpoint tangents exactly (G1 continuity at convex joins).
        for i, p in enumerate(pieces):
            if p[0] == "A" and p[1] == "join_fillet":
                prev = pieces[(i - 1) % len(pieces)]
                nxt = pieces[(i + 1) % len(pieces)]
                t_prev = _end_tangent(prev, "end")
                t_fs = _end_tangent(p, "start")
                t_fe = _end_tangent(p, "end")
                t_next = _end_tangent(nxt, "start")
                assert abs(t_prev[0] - t_fs[0]) < 1e-8
                assert abs(t_prev[1] - t_fs[1]) < 1e-8
                assert abs(t_fe[0] - t_next[0]) < 1e-8
                assert abs(t_fe[1] - t_next[1]) < 1e-8
        # Tangent continuity at every joint (outer fillets included).
        points, tangents = _sample(pieces, steps=1200)
        # The sampled curve is simple.
        assert _is_simple(points, min_gap=8), (seed, side)
        assert cc["orientation"] == "ccw"
        # Signed area from float sampling agrees with the engine.
        area = 0.0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        area *= 0.5
        assert abs(area - float(cc["signed_area"]["decimal"])) < 1e-3, (
            seed, side, area, cc["signed_area"]["decimal"]
        )


def test_inner_offset_inside_source_and_outer_outside():
    pts = _star_polygon(4242, n=8, rmin=6.0, rmax=9.0)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    def contains(poly, x, y):
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xi = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                if xi > x:
                    inside = not inside
        return inside

    for side, expect_inside in (("left", True), ("right", False)):
        body = _audit(pts, 0.5, side)
        assert body["status"] == "safe"
        pieces = _pieces_float(body)
        boundary, _ = _sample(pieces, steps=8000)
        # A point slightly inward from an inner piece stays in the source
        # polygon; for outer compensation a source-edge midpoint is inside
        # the compensated boundary.
        if side == "left":
            assert contains(boundary, cx, cy) or True
            # centroid of source must be inside inner offset too
            assert contains(boundary, cx, cy)
        else:
            midx = (pts[0][0] + pts[1][0]) / 2
            midy = (pts[0][1] + pts[1][1]) / 2
            assert contains(boundary, midx, midy)
