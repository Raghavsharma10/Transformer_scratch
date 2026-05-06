def solve_let(expr, vars):
    """Solves a let-form by calling RHS with nested scope."""
    lhs_value = solve(expr.lhs, vars).value
    if not isinstance(lhs_value, structured.IStructured):
        raise errors.EfilterTypeError(
            root=expr.lhs, query=expr.original,
            message="The LHS of 'let' must evaluate to an IStructured. Got %r."
            % (lhs_value,))

    return solve(expr.rhs, __nest_scope(expr.lhs, vars, lhs_value))