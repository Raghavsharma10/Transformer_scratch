def line_evaluate(t, p0, a, b, c):
    """Evaluate the orthogonal polynomial defined by its recurrence coefficients
    a, b, and c at the point(s) t.
    """
    vals1 = numpy.zeros_like(t, dtype=int)
    # The order is important here; see
    # <https://github.com/sympy/sympy/issues/13637>.
    vals2 = numpy.ones_like(t) * p0

    for a_k, b_k, c_k in zip(a, b, c):
        vals0, vals1 = vals1, vals2
        vals2 = vals1 * (t * a_k - b_k) - vals0 * c_k
    return vals2