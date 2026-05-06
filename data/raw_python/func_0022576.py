def normalize(expr):
    """Normalize both sides, but don't eliminate the expression."""
    lhs = normalize(expr.lhs)
    rhs = normalize(expr.rhs)
    return type(expr)(lhs, rhs, start=lhs.start, end=rhs.end)