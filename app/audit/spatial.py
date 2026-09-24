"""Exact spatial index for pair pruning.

The grid cell size and indices are *exact*: a decimal display value is used
only as an initial guess for the cell index and is corrected with exact
field comparisons against integer multiples of the (rational) cell size.
Consequently the index can only merge or duplicate candidate pairs, never
silently drop a genuinely intersecting pair.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Dict, Iterator, List, Set, Tuple

from ..exact.algebraic import Field
from ..exact.encode import evaluate_decimal
from .primitives import Box


def _exact_floor(F: Field, value, cell: Fraction, origin) -> int:
    """floor((value - origin) / cell), with cell, origin rational elements."""
    cell_el = F.rational(cell)
    rel = F.sub(value, origin)
    ratio_dec = evaluate_decimal(F, F.div(rel, cell_el), 6)
    k = int(ratio_dec.to_integral_value(rounding="ROUND_FLOOR"))
    # Exact correction around the decimal guess.
    while not (F.le(F.rational(k * cell), rel)
               and F.lt(rel, F.rational((k + 1) * cell))):
        if F.lt(rel, F.rational(k * cell)):
            k -= 1
        else:
            k += 1
    return k


class ExactGrid:
    def __init__(self, F: Field, boxes: List[Box]):
        self.F = F
        self.boxes = boxes
        # Rational cell size: the largest box span, rounded up to an integer
        # number of thousandths, at least 1.
        max_span = F.rational(1)
        for b in boxes:
            w = F.sub(b.xmax, b.xmin)
            h = F.sub(b.ymax, b.ymin)
            if F.gt(w, max_span):
                max_span = w
            if F.gt(h, max_span):
                max_span = h
        dec = evaluate_decimal(F, max_span, 9)
        cell = Fraction(int((dec + Decimal("0.000001")) * 1000), 1000)
        if cell <= 0:
            cell = Fraction(1)
        self.cell = cell
        # Rational origin: floor of the global minimum at cell precision.
        gmin_x = boxes[0].xmin
        gmin_y = boxes[0].ymin
        for b in boxes[1:]:
            if F.lt(b.xmin, gmin_x):
                gmin_x = b.xmin
            if F.lt(b.ymin, gmin_y):
                gmin_y = b.ymin
        self.ox = F.rational(
            cell * _exact_floor(F, gmin_x, cell, F.rational(0))
        )
        self.oy = F.rational(
            cell * _exact_floor(F, gmin_y, cell, F.rational(0))
        )
        self.cells: Dict[Tuple[int, int], List[int]] = {}
        for i, b in enumerate(boxes):
            ix0 = _exact_floor(F, b.xmin, cell, self.ox)
            ix1 = _exact_floor(F, b.xmax, cell, self.ox)
            iy0 = _exact_floor(F, b.ymin, cell, self.oy)
            iy1 = _exact_floor(F, b.ymax, cell, self.oy)
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    self.cells.setdefault((ix, iy), []).append(i)

    def candidate_pairs(self) -> Iterator[Tuple[int, int]]:
        seen: Set[Tuple[int, int]] = set()
        for members in self.cells.values():
            members.sort()
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    pair = (members[a], members[b])
                    if pair not in seen:
                        seen.add(pair)
                        yield pair
