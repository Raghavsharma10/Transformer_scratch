def augknt(knots, order):
    """Augment a knot vector.

Parameters:
    knots:
        Python list or rank-1 array, the original knot vector (without endpoint repeats)
    order:
        int, >= 0, order of spline

Returns:
    list_of_knots:
        rank-1 array that has (`order` + 1) copies of ``knots[0]``, then ``knots[1:-1]``, and finally (`order` + 1) copies of ``knots[-1]``.

Caveats:
    `order` is the spline order `p`, not `p` + 1, and existing knots are never deleted.
    The knot vector always becomes longer by calling this function.
"""
    if isinstance(knots, np.ndarray)  and  knots.ndim > 1:
        raise ValueError("knots must be a list or a rank-1 array")
    knots = list(knots)  # ensure Python list

    # One copy of knots[0] and knots[-1] will come from "knots" itself,
    # so we only need to prepend/append "order" copies.
    #
    return np.array( [knots[0]] * order  +  knots  +  [knots[-1]] * order )