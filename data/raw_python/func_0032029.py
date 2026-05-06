def mode(data):
    """Return the most common data point from discrete or nominal data.

    ``mode`` assumes discrete data, and returns a single value. This is the
    standard treatment of the mode as commonly taught in schools:

    If there is not exactly one most common value, ``mode`` will raise
    StatisticsError.
    """
    # Generate a table of sorted (value, frequency) pairs.
    table = counts(data)
    if len(table) == 1:
        return table[0][0]
    elif table:
        raise StatisticsError(
            'no unique mode; found %d equally common values' % len(table)
        )
    else:
        raise StatisticsError('no mode for empty data')