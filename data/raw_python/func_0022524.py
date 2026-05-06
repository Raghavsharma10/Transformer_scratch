def solve_filter(expr, vars):
    """Filter values on the LHS by evaluating RHS with each value.

    Returns any LHS values for which RHS evaluates to a true value.
    """
    lhs_values, _ = __solve_for_repeated(expr.lhs, vars)

    def lazy_filter():
        for lhs_value in repeated.getvalues(lhs_values):
            if solve(expr.rhs, __nest_scope(expr.lhs, vars, lhs_value)).value:
                yield lhs_value

    return Result(repeated.lazy(lazy_filter), ())