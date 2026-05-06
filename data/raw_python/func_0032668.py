def _parse_values(values, extra=None):
    """
    Utility function to flatten out args.

    For internal use only.

    :param values: list, tuple, or str
    :param extra: list or None
    :return: list
    """
    coerced = list(values)

    if coerced == values:
        values = coerced
    else:
        coerced = tuple(values)
        if coerced == values:
            values = list(values)
        else:
            values = [values]

    if extra:
        values.extend(extra)
    return values