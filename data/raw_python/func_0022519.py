def solve_repeat(expr, vars):
    """Build a repeated value from subexpressions."""
    try:
        result = repeated.meld(*[solve(x, vars).value for x in expr.children])
        return Result(result, ())
    except TypeError:
        raise errors.EfilterTypeError(
            root=expr, query=expr.source,
            message="All values in a repeated value must be of the same type.")