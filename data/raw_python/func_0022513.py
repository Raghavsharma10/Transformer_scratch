def __solve_and_destructure_repeated(expr, vars):
    """Helper: solve 'expr' always returning a list of scalars.

    If the output of 'expr' is one or more row tuples with only a single column
    then return a repeated value of values in that column. If there are more
    than one column per row then raise.

    This returns a list because there's no point in wrapping the scalars in
    a repeated value for use internal to the implementing solver.

    Returns:
        Two values:
         - An iterator (not an IRepeated!) of scalars.
         - A boolean to indicate whether the original value was repeating.

    Raises:
        EfilterTypeError if the values don't conform.
    """
    iterable, isrepeating = __solve_for_repeated(expr, vars)
    if iterable is None:
        return (), isrepeating

    if not isrepeating:
        return [iterable], False

    values = iter(iterable)

    try:
        value = next(values)
    except StopIteration:
        return (), True

    if not isinstance(value, row_tuple.RowTuple):
        result = [value]
        # We skip type checking the remaining values because it'd be slow.
        result.extend(values)
        return result, True

    try:
        result = [value.get_singleton()]
        for value in values:
            result.append(value.get_singleton())

        return result, True
    except ValueError:
        raise errors.EfilterTypeError(
            root=expr, query=expr.source,
            message="Was expecting exactly one column in %r." % (value,))