def normalize(value, series, offset=0):
    r"""
    Scale a value to the range defined by a series.

    :param value: Value to normalize
    :type  value: number

    :param series: List of numbers that defines the normalization range
    :type  series: list

    :param offset: Normalization offset, i.e. the returned value will be in
                   the range [**offset**, 1.0]
    :type  offset: number

    :rtype: number

    :raises:
     * RuntimeError (Argument \`offset\` is not valid)

     * RuntimeError (Argument \`series\` is not valid)

     * RuntimeError (Argument \`value\` is not valid)

     * ValueError (Argument \`offset\` has to be in the [0.0, 1.0] range)

     * ValueError (Argument \`value\` has to be within the bounds of the
       argument \`series\`)

    For example::

        >>> import pmisc
        >>> pmisc.normalize(15, [10, 20])
        0.5
        >>> pmisc.normalize(15, [10, 20], 0.5)
        0.75
    """
    if not _isreal(value):
        raise RuntimeError("Argument `value` is not valid")
    if not _isreal(offset):
        raise RuntimeError("Argument `offset` is not valid")
    try:
        smin = float(min(series))
        smax = float(max(series))
    except:
        raise RuntimeError("Argument `series` is not valid")
    value = float(value)
    offset = float(offset)
    if not 0 <= offset <= 1:
        raise ValueError("Argument `offset` has to be in the [0.0, 1.0] range")
    if not smin <= value <= smax:
        raise ValueError(
            "Argument `value` has to be within the bounds of argument `series`"
        )
    return offset + ((1.0 - offset) * (value - smin) / (smax - smin))