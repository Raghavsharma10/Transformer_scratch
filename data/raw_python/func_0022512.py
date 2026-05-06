def __solve_for_scalar(expr, vars):
    """Helper: solve 'expr' always returning a scalar (not IRepeated).

    If the output of 'expr' is a single value or a single RowTuple with a single
    column then return the value in that column. Otherwise raise.

    Arguments:
        expr: Expression to solve.
        vars: The scope.

    Returns:
        A scalar value (not an IRepeated).

    Raises:
        EfilterTypeError if it cannot get a scalar.
    """
    var = solve(expr, vars).value
    try:
        scalar = repeated.getvalue(var)
    except TypeError:
        raise errors.EfilterTypeError(
            root=expr, query=expr.source,
            message="Wasn't expecting more than one value here. Got %r."
            % (var,))

    if isinstance(scalar, row_tuple.RowTuple):
        try:
            return scalar.get_singleton()
        except ValueError:
            raise errors.EfilterTypeError(
                root=expr, query=expr.source,
                message="Was expecting a scalar value here. Got %r."
                % (scalar,))
    else:
        return scalar