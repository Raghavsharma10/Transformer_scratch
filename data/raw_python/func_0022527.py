def solve_cast(expr, vars):
    """Get cast LHS to RHS."""
    lhs = solve(expr.lhs, vars).value
    t = solve(expr.rhs, vars).value

    if t is None:
        raise errors.EfilterTypeError(
            root=expr, query=expr.source,
            message="Cannot find type named %r." % expr.rhs.value)

    if not isinstance(t, type):
        raise errors.EfilterTypeError(
            root=expr.rhs, query=expr.source,
            message="%r is not a type and cannot be used with 'cast'." % (t,))

    try:
        cast_value = t(lhs)
    except TypeError:
        raise errors.EfilterTypeError(
            root=expr, query=expr.source,
            message="Invalid cast %s -> %s." % (type(lhs), t))

    return Result(cast_value, ())