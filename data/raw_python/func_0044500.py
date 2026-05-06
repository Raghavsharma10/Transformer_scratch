def mode(data):
    """Return the most common data point from discrete or nominal data.

    ``mode`` assumes discrete data, and returns a single value. This is the
    standard treatment of the mode as commonly taught in schools:

    >>> mode([1, 1, 2, 3, 3, 3, 3, 4])
    3

    This also works with nominal (non-numeric) data:

    >>> mode(["red", "blue", "blue", "red", "green", "red", "red"])
    'red'
    """

    # Generate a table of sorted (value, frequency) pairs.
    hist = collections.Counter(data)
    top = hist.most_common(2)

    if len(top) == 1:
        return top[0][0]
    elif not top:
        raise StatisticsError('no mode for empty data')
    elif top[0][1] == top[1][1]:
        raise StatisticsError(
            'no unique mode; found %d equally common values' % len(hist)
            )
    else:
        return top[0][0]