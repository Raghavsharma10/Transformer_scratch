def solve_isinstance(expr, vars):
    """Typecheck whether LHS is type on the RHS."""
    lhs = solve(expr.lhs, vars)

    try:
        t = solve(expr.rhs, vars).value
    except errors.EfilterKeyError:
        t = None

    if t is None:
        raise errors.EfilterTypeError(
            root=expr.rhs, query=expr.source,
            message="Cannot find type named %r." % expr.rhs.value)

    if not isinstance(t, type):
        raise errors.EfilterTypeError(
            root=expr.rhs, query=expr.source,
            message="%r is not a type and cannot be used with 'isa'." % (t,))

    return Result(protocol.implements(lhs.value, t), ())