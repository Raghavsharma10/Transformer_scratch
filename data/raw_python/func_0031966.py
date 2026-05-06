def closed(lower_value, upper_value):
    """Helper function to construct an interval object with closed lower and upper.

    For example:

    >>> closed(100.2, 800.9)
    [100.2, 800.9]
    """
    return Interval(Interval.CLOSED, lower_value, upper_value, Interval.CLOSED)