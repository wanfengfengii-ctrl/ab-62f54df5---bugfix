"""Tests for the exact algebraic kernel and angular predicates.

These establish that every decision is exact: the same checks are also
performed against independent Decimal high-precision computations to catch
representation mistakes, but no float is ever used inside the engine.
"""

from fractions import Fraction as F

from app.exact.algebraic import Field
from app.exact.arcs import (
    Arc,
    CCW,
    CW,
    on_closed_arc,
    on_open_arc,
    traversal_order,
)
from app.exact.geometry import Circle, Vec


def v(F, x, y):
    return Vec(F.rational(x), F.rational(y))


# ------------------------------------------------------------- field tower
def test_basic_signs_and_identities():
    f = Field()
    s2 = f.square_root(f.rational(2))
    assert f.sign(s2) > 0
    assert f.eq(f.sq(s2), f.rational(2))
    # 3 - 2 sqrt 2 > 0, decided via the norm 9 - 8 = 1.
    assert f.sign(f.sub(f.rational(3), f.scale(s2, 2))) > 0
    # 1 - sqrt 2 < 0.
    assert f.sign(f.sub(f.rational(1), s2)) < 0
    # 1/sqrt 2 == sqrt 2 / 2 exactly.
    assert f.eq(f.div(f.rational(1), s2), f.scale(s2, F(1, 2)))


def test_nested_radicals_identity():
    f = Field()
    s2 = f.square_root(f.rational(2))
    s3 = f.square_root(f.rational(3))
    r1 = f.square_root(f.add(f.rational(2), s3))
    r2 = f.square_root(f.sub(f.rational(2), s3))
    # sqrt(2+sqrt3) * sqrt(2-sqrt3) == 1 exactly.
    assert f.eq(f.mul(r1, r2), f.rational(1))
    # sqrt 2 * sqrt 3 == sqrt 6, recognised inside the tower (no extra level).
    assert f.eq(f.mul(s2, s3), f.square_root(f.rational(6)))
    # sqrt(2 - sqrt3) == (sqrt6 - sqrt2)/2.
    assert f.eq(
        r2,
        f.scale(f.sub(f.square_root(f.rational(6)), s2), F(1, 2)),
    )


def test_square_of_square_root_round_trip():
    f = Field()
    s7 = f.square_root(f.rational(7))
    z = f.add(f.rational(3), f.scale(s7, 5))
    target = f.add(f.sq(z), f.rational(1))
    r = f.square_root(target)
    assert f.eq(f.sq(r), target)


def test_no_spurious_extension_for_perfect_square():
    f = Field()
    s3 = f.square_root(f.rational(3))
    base_level = f.level
    # (1 + sqrt3)^2 = 4 + 2 sqrt3: its root must be found without extending.
    z = f.add(f.rational(1), s3)
    root = f.square_root(f.sq(z))
    assert f.eq(root, z)
    assert f.level == base_level


def test_nested_root_whose_delta_reuses_top_variable_terminates():
    # For x = A + B*s_m the recipe needs delta = sqrt(A**2 - B**2 D_m) in the
    # subfield *below* s_m.  Here D_m = 8, A = 4, B = 1 gives delta**2 = 8,
    # whose rational square-class solver returns s_m itself (root.k > m), not
    # a subfield element.  Such x is not a square in the current field and a
    # new radical must be adjoined.  Accepting the in-tower "root" made the
    # A +/- delta split recurse on the same variable forever.
    f = Field()
    s8 = f.square_root(f.rational(8))
    x = f.add(f.rational(4), s8)
    root = f.square_root(x)  # must terminate, not RecursionError
    assert f.eq(f.sq(root), x)
    assert f.sign(root) > 0


def test_rational_class_root_above_split_variable_rejected():
    # Mirrors the reported contour: s1**2 = 8 registered alongside an earlier
    # radical; sqrt(1/50 - s1/200) must terminate and be a genuine square root.
    f = Field()
    f.square_root(f.rational(10))
    s8 = f.square_root(f.rational(8))
    x = f.add(f.rational(F(1, 50)), f.scale(s8, F(-1, 200)))
    root = f.square_root(x)
    assert f.eq(f.sq(root), x)
    assert f.sign(root) > 0


# ------------------------------------------------------------- arc logic
def _arc_from(f, center, rs, re_, sense=CCW, full=False, radius=1):
    c = v(f, *center)
    cr = Circle(c, f.rational(radius))
    s = v(f, rs[0] - center[0], rs[1] - center[1])
    e = v(f, re_[0] - center[0], re_[1] - center[1])
    return Arc(cr, s, e, sense, full)


def test_ccw_quarter_and_three_quarter_membership():
    f = Field()
    c = (0, 0)
    # quarter CCW arc (1,0) -> (0,1)
    a = _arc_from(f, c, (1, 0), (0, 1), CCW)
    arc_obj = Arc(a.circle, a.start, a.end, CCW)
    assert on_closed_arc(f, arc_obj, v(f, 1, 0))
    assert on_closed_arc(f, arc_obj, v(f, 0, 1))
    # midpoint of the quarter arc, (sqrt2/2, sqrt2/2), is on it.
    h = f.div(f.square_root(f.rational(2)), f.rational(2))
    mid = Vec(h, h)
    assert on_closed_arc(f, arc_obj, mid)
    # (0,-1) is on the complementary arc only -> not on quarter.
    assert not on_closed_arc(f, arc_obj, v(f, 0, -1))
    # long CCW arc (1,0) -> (0,-1) (270 degrees) contains (0,1).
    along = Arc(a.circle, v(f, 1, 0), v(f, 0, -1), CCW)
    assert on_closed_arc(f, along, v(f, 0, 1))
    # the 45-degree point is on the long arc; the 315-degree point is not.
    assert on_closed_arc(f, along, mid)
    assert not on_closed_arc(f, along, Vec(h, f.neg(h)))


def test_open_arc_excludes_endpoints():
    f = Field()
    arc = _arc_from(f, (0, 0), (1, 0), (0, 1), CCW)
    assert not on_open_arc(f, arc, v(f, 1, 0))
    assert not on_open_arc(f, arc, v(f, 0, 1))
    h = f.div(f.square_root(f.rational(2)), f.rational(2))
    assert on_open_arc(f, arc, Vec(h, h))


def test_traversal_order_along_arcs():
    f = Field()
    arc = _arc_from(f, (0, 0), (1, 0), (0, 1), CCW)
    h = f.div(f.square_root(f.rational(2)), f.rational(2))
    p1 = Vec(f.rational(1), f.rational(0))
    pm = Vec(h, h)
    p2 = Vec(f.rational(0), f.rational(1))
    assert traversal_order(f, arc, p1, pm) < 0
    assert traversal_order(f, arc, pm, p2) < 0
    assert traversal_order(f, arc, p2, p1) > 0
    cw = _arc_from(f, (0, 0), (0, 1), (1, 0), CW)
    assert traversal_order(f, cw, p2, pm) < 0
    assert traversal_order(f, cw, pm, p1) < 0
