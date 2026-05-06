def solve_sort(expr, vars):
    """Sort values on the LHS by the value they yield when passed to RHS."""
    lhs_values = repeated.getvalues(__solve_for_repeated(expr.lhs, vars)[0])

    sort_expression = expr.rhs

    def _key_func(x):
        return solve(sort_expression, __nest_scope(expr.lhs, vars, x)).value

    results = ordered.ordered(lhs_values, key_func=_key_func)

    return Result(repeated.meld(*results), ())