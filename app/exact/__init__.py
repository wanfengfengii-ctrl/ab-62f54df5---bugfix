"""Exact geometry kernel built on decimal inputs.

No floating point is used for geometric decisions.  Decimal input values are
converted to rationals (:mod:`app.exact.number`) and every algebraic quantity
encountered (circle intersections, tangencies) is represented exactly in a
nested quadratic-field tower (:mod:`app.exact.algebraic`).
"""
