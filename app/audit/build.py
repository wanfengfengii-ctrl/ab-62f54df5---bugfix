"""Exact construction of the compensated contour.

Three exact operations are used:

1. **Line translation** -- a source line moves by ``tau * tool`` along its
   unit normal (left for ``tau=+1``), keeping direction and length.
2. **Concentric arc re-radius** -- a source arc keeps center and sense; the
   offset radius is ``rho - tool`` (tool facing the center; collapse and
   inversion are decided exactly by comparing ``rho**2`` with ``tool**2``
   while both are still rational) or ``rho + tool``.
3. **Vertex joins** -- with unit tangents ``tA`` (incoming) and ``tB``
   (outgoing) and ``c = sign(cross(tA, tB))``:

   * ``c * tau < 0`` (convex for the tool): a prescribed radius-``tool``
     fillet centered on the original vertex joins the two exact tangent
     points, sweeping the short arc in sense ``c`` (type-B round join);
   * ``c * tau > 0`` (concave for the tool): the two offset supports are
     extended/trimmed to their exact intersection (line-line, line-circle or
     circle-circle, algebraically);
   * ``c == 0``: tangentially continuous (single shared tangent point) or a
     traversal reversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple

from ..exact.algebraic import Element, Field
from ..exact.geometry import (
    Circle,
    Line,
    Vec,
    circle_circle_intersections,
    left_of,
    line_circle_intersections,
    line_line_intersection,
    right_of,
    vadd,
    vcross,
    vdot,
    vscale,
    vsub,
)
from .primitives import ArcP, Seg, unit_tangent_arc_at, unit_tangent_line

LEFT = 1
RIGHT = -1


@dataclass
class SourceEnt:
    index: int
    kind: str  # 'line' | 'arc'
    start: Tuple[Fraction, Fraction]
    end: Tuple[Fraction, Fraction]
    center: Optional[Tuple[Fraction, Fraction]]
    sense: int  # +1/-1 for arcs, 0 for lines
    full: bool
    tau: int
    label: Optional[str]
    radius2: Fraction = Fraction(0)


@dataclass
class JoinResolution:
    kind: str  # 'smooth' | 'fillet' | 'intersection' | 'reversal'
    point: Optional[Vec] = None  # intersection point for sharp joins
    sense: int = 0  # fillet sweep sense


class BuildError(Exception):
    def __init__(self, code: str, message: str, sources: List[int], **extra):
        super().__init__(message)
        self.code = code
        self.message = message
        self.sources = sources
        self.extra = extra


class OffsetBuilder:
    def __init__(self, F: Field, ents: List[SourceEnt], tool: Fraction):
        self.F = F
        self.ents = ents
        self.tool = tool
        self.T = F.rational(tool)
        self.n = len(ents)
        # Exact offset supports and nominal tangent points.
        self.sup_line: List[Optional[Line]] = [None] * self.n
        self.sup_circle: List[Optional[Circle]] = [None] * self.n
        self.sup_sense: List[int] = [0] * self.n
        self.js: List[Optional[Vec]] = [None] * self.n
        self.je: List[Optional[Vec]] = [None] * self.n
        self.ts: List[Optional[Vec]] = [None] * self.n
        self.te: List[Optional[Vec]] = [None] * self.n
        self.comp_r: List[Optional[Element]] = [None] * self.n
        self.trim_s: List[Optional[Vec]] = [None] * self.n
        self.trim_e: List[Optional[Vec]] = [None] * self.n

    # ------------------------------------------------------------ utilities
    def pt(self, q: Tuple[Fraction, Fraction]) -> Vec:
        return Vec(self.F.rational(q[0]), self.F.rational(q[1]))

    def _eq(self, a: Vec, b: Vec) -> bool:
        return self.F.eq(a.x, b.x) and self.F.eq(a.y, b.y)

    def side_normal(self, t: Vec, tau: int) -> Vec:
        n = left_of(self.F, t) if tau > 0 else right_of(self.F, t)
        return n

    def tangent_at(self, i: int, p: Vec) -> Vec:
        if self.ents[i].kind == "line":
            return self.sup_line[i].u
        return unit_tangent_arc_at(self.F, self.sup_circle[i].c, p, self.sup_sense[i])

    # ------------------------------------------------- radius collapse checks
    def check_radii(self) -> None:
        t2 = self.tool * self.tool
        for e in self.ents:
            if e.kind != "arc":
                continue
            if e.sense == e.tau and e.radius2 <= t2:
                if e.radius2 == t2:
                    raise BuildError(
                        "radius_collapse",
                        f"entity {e.index}: compensated arc radius "
                        f"rho - tool collapses exactly to zero",
                        [e.index],
                        source_radius_squared=self.F.rational(e.radius2),
                        tool_radius=self.T,
                    )
                raise BuildError(
                    "radius_inversion",
                    f"entity {e.index}: tool radius {self.tool} is larger "
                    f"than arc radius sqrt({e.radius2}); the concentric "
                    "offset would invert its orientation",
                    [e.index],
                    source_radius_squared=self.F.rational(e.radius2),
                    tool_radius=self.T,
                )

    # ------------------------------------------------------------- supports
    def build_supports(self) -> None:
        F = self.F
        for i, e in enumerate(self.ents):
            sp, ep = self.pt(e.start), self.pt(e.end)
            tau = e.tau
            if e.kind == "line":
                u = unit_tangent_line(F, sp, ep)
                self.ts[i] = u
                self.te[i] = u
                n = self.side_normal(u, tau)
                shift = vscale(F, n, self.T)
                p0 = vadd(F, sp, shift)
                p1 = vadd(F, ep, shift)
                self.sup_line[i] = Line(p0, u)
                self.js[i], self.je[i] = p0, p1
                self.sup_sense[i] = 0
                self.comp_r[i] = None
            else:
                cp = self.pt(e.center)
                rho = F.square_root(F.rational(e.radius2))
                ts = unit_tangent_arc_at(F, cp, sp, e.sense)
                te = unit_tangent_arc_at(F, cp, ep, e.sense)
                self.ts[i], self.te[i] = ts, te
                rcomp = (
                    F.sub(rho, self.T) if e.sense == tau else F.add(rho, self.T)
                )
                self.comp_r[i] = rcomp
                self.sup_circle[i] = Circle(cp, rcomp)
                self.sup_sense[i] = e.sense
                # Tangent points V + tool * n_comp(t) land on the offset
                # circle exactly (concentric re-radius identity).
                self.js[i] = vadd(F, sp, vscale(F, self.side_normal(ts, tau), self.T))
                self.je[i] = vadd(F, ep, vscale(F, self.side_normal(te, tau), self.T))

    # -------------------------------------------- forward ordering on supports
    def _line_param(self, i: int, p: Vec) -> Element:
        ln = self.sup_line[i]
        return vdot(self.F, vsub(self.F, p, ln.p), ln.u)

    def forward(self, i: int, a: Vec, b: Vec) -> bool:
        """Strictly forward from a to b on entity i's bounded piece.

        For arcs "forward" means along the actual swept arc between the
        entity's nominal tangent points (never the complementary arc); a
        and b must both lie on that bounded arc.
        """
        F = self.F
        if self.ents[i].kind == "line":
            return F.gt(self._line_param(i, b), self._line_param(i, a))
        from ..exact.arcs import traversal_order

        ea = self._bounded_arc(i)
        return traversal_order(F, ea, a, b) < 0

    def _bounded_arc(self, i: int):
        from ..exact.arcs import Arc as EArc

        return EArc(
            self.sup_circle[i],
            vsub(self.F, self.js[i], self.sup_circle[i].c),
            vsub(self.F, self.je[i], self.sup_circle[i].c),
            self.sup_sense[i],
            full=False,
        )

    def on_bounded(self, i: int, p: Vec) -> bool:
        """p lies on the closed bounded nominal piece of entity i."""
        if self.ents[i].kind == "line":
            from .intersections import seg_contains
            from .primitives import Seg

            return seg_contains(
                self.F, Seg(self.js[i], self.je[i], self.sup_line[i].u), p
            )
        from ..exact.arcs import on_closed_arc

        return on_closed_arc(self.F, self._bounded_arc(i), p)

    # --------------------------------------------------------- intersections
    def support_intersections(self, i: int, j: int) -> Tuple[List[Vec], str]:
        """All intersections of complete supports i, j; (points, relation)."""
        F = self.F
        if self.ents[i].kind == "line" and self.ents[j].kind == "line":
            _, p, coinc = line_line_intersection(F, self.sup_line[i], self.sup_line[j])
            if coinc:
                return [], "coincident_line"
            return ([p], "cross") if p is not None else ([], "disjoint")
        if self.ents[i].kind == "line":
            return self._line_circle(i, j)
        if self.ents[j].kind == "line":
            pts, rel = self._line_circle(j, i)
            return pts, rel
        pts, rel = circle_circle_intersections(F, self.sup_circle[i], self.sup_circle[j])
        return [p for p, _ in pts], rel

    def _line_circle(self, li: int, ci: int) -> Tuple[List[Vec], str]:
        F = self.F
        raw = line_circle_intersections(F, self.sup_line[li], self.sup_circle[ci])
        if not raw:
            return [], "disjoint"
        rel = "tangent" if len(raw) == 1 else "cross"
        return [p for _, p, _ in raw], rel

    # ------------------------------------------------------------- per-vertex
    def resolve_vertex(self, k: int) -> JoinResolution:
        """Join incoming entity k-1 with outgoing entity k at vertex k."""
        F = self.F
        i, j = (k - 1) % self.n, k % self.n
        a, b = self.ents[i], self.ents[j]
        tA, tB = self.te[i], self.ts[j]
        JA, JB = self.je[i], self.js[j]
        cr = vcross(F, tA, tB)
        dt = vdot(F, tA, tB)
        c, d = F.sign(cr), F.sign(dt)
        if a.tau != b.tau:
            raise BuildError(
                "compensation_side_switch",
                f"vertex {k}: entities {a.index} and {b.index} request "
                "different compensation sides",
                [a.index, b.index],
                join_index=k,
            )
        tau = a.tau
        if c == 0:
            if d > 0:
                if not (F.eq(JA.x, JB.x) and F.eq(JA.y, JB.y)):
                    raise BuildError(
                        "tangent_join_mismatch",
                        f"vertex {k}: tangents agree but the exact tangent "
                        "points differ",
                        [a.index, b.index],
                        join_index=k,
                        point_a=JA,
                        point_b=JB,
                    )
                return JoinResolution("smooth", point=JA)
            raise BuildError(
                "traversal_reversal",
                f"vertex {k}: the path reverses direction (exactly opposite "
                "tangents); no orientation-preserving compensation exists",
                [a.index, b.index],
                join_index=k,
                point=self.pt(b.start),
            )
        if c * tau < 0:
            # Convex for the tool: prescribed round join.  Traversal
            # tangency forces the fillet sense to -tau; the fillet condition
            # c*tau < 0 guarantees it sweeps the short arc.
            return JoinResolution("fillet", sense=-tau)
        # Concave: extend/trim supports to their exact intersection.
        pts, rel = self.support_intersections(i, j)
        if rel in ("coincident_line", "coincident_circle"):
            raise BuildError(
                "coincident_supports",
                f"vertex {k}: offset supports of entities {a.index} and "
                f"{b.index} coincide exactly",
                [a.index, b.index],
                join_index=k,
            )
        chosen = self._select_concave_intersection(i, j, pts, k)
        return JoinResolution("intersection", point=chosen)

    def _select_concave_intersection(self, i: int, j: int, pts, k: int):
        """Pick the unique physically correct concave join root.

        A line and a circle (or two circles) can meet twice.  The join must
        simultaneously be the *last* reachable intersection along the
        incoming bounded piece and the *first* along the outgoing bounded
        piece; reachability is membership in the entity's nominal bounded
        offset piece.  If no single point satisfies both, the tool gouges.
        """
        incoming = [p for p in pts if self.on_bounded(i, p)]
        outgoing = [p for p in pts if self.on_bounded(j, p)]
        if not incoming or not outgoing:
            raise BuildError(
                "join_intersection_missing",
                f"vertex {k}: the concave offset supports of entities "
                f"{self.ents[i].index} and {self.ents[j].index} do not meet "
                "at a reachable exact intersection",
                [self.ents[i].index, self.ents[j].index],
                join_index=k,
                tangent_point_a=self.je[i],
                tangent_point_b=self.js[j],
            )

        c_in = incoming[0]
        for p in incoming[1:]:
            if self.forward(i, c_in, p):  # p is later along incoming
                c_in = p
        c_out = outgoing[0]
        for p in outgoing[1:]:
            if self.forward(j, p, c_out):  # p is earlier along outgoing
                c_out = p
        if not self._eq(c_in, c_out):
            raise BuildError(
                "gouging_corner",
                f"vertex {k}: the offset supports of entities "
                f"{self.ents[i].index} and {self.ents[j].index} do not share "
                "a single concave join point (the tool gouges the corner)",
                [self.ents[i].index, self.ents[j].index],
                join_index=k,
                point_a=c_in,
                point_b=c_out,
            )
        return c_in

    def _check_trim_order(self, joins) -> None:
        """Each entity must be traversed forward between its two join points.

        When the tool is too large for a concave region the exact join
        intersections lie past one another, so the trimmed piece would run
        backwards; that is an orientation-destroying interference and is
        rejected (it would later self-intersect anyway).
        """
        for i, e in enumerate(self.ents):
            s, z = self.trim_s[i], self.trim_e[i]
            if self.F.eq(s.x, z.x) and self.F.eq(s.y, z.y):
                continue
            if not self.forward(i, s, z):
                raise BuildError(
                    "trim_order_reversal",
                    f"entity {e.index}: the required trim points are in "
                    "reverse traversal order -- the tool cannot fit the "
                    "adjacent concave region without gouging",
                    [e.index],
                    start_trim=s,
                    end_trim=z,
                )

    # ----------------------------------------------------------------- build
    def build(self) -> Tuple[List, List[JoinResolution]]:
        self.check_radii()
        self.build_supports()
        joins: List[JoinResolution] = []
        for k in range(self.n):
            joins.append(self.resolve_vertex(k))
        # Apply trims from concave (intersection) and smooth joins.
        for k, res in enumerate(joins):
            i, j = (k - 1) % self.n, k % self.n
            if res.kind == "intersection":
                self.trim_e[i] = res.point
                self.trim_s[j] = res.point
            elif res.kind == "smooth":
                self.trim_e[i] = res.point
                self.trim_s[j] = res.point
        for i in range(self.n):
            if self.trim_s[i] is None:
                self.trim_s[i] = self.js[i]
            if self.trim_e[i] is None:
                self.trim_e[i] = self.je[i]
        self._check_trim_order(joins)
        chain: List = []
        for i, e in enumerate(self.ents):
            chain.append(self._piece(i))
            res = joins[(i + 1) % self.n]
            if res.kind == "fillet":
                chain.append(self._fillet((i + 1) % self.n, res))
        return chain, joins

    def _piece(self, i: int) -> object:
        e = self.ents[i]
        s, z = self.trim_s[i], self.trim_e[i]
        if e.kind == "line":
            return Seg(s, z, self.sup_line[i].u, source=i)
        return ArcP(
            self.sup_circle[i],
            s,
            z,
            sense=self.sup_sense[i],
            full=e.full,
            kind="offset_arc",
            source=i,
        )

    def _fillet(self, k: int, res: JoinResolution) -> ArcP:
        i = (k - 1) % self.n
        V = self.pt(self.ents[k].start)
        return ArcP(
            Circle(V, self.T),
            self.je[i],
            self.js[k],
            sense=res.sense,
            full=False,
            kind="join_fillet",
            source=None,
            join_vertex=k,
        )
