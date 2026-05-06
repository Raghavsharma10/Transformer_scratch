def solve_each(expr, vars):
    """Return True if RHS evaluates to a true value with each state of LHS.

    If LHS evaluates to a normal IAssociative object then this is the same as
    a regular let-form, except the return value is always a boolean. If LHS
    evaluates to a repeared var (see efilter.protocols.repeated) of
    IAssociative objects then RHS will be evaluated with each state and True
    will be returned only if each result is true.
    """
    lhs_values, _ = __solve_for_repeated(expr.lhs, vars)

    for lhs_value in repeated.getvalues(lhs_values):
        result = solve(expr.rhs, __nest_scope(expr.lhs, vars, lhs_value))
        if not result.value:
            # Each is required to return an actual boolean.
            return result._replace(value=False)

    return Result(True, ())