def closedopen(lower_value, upper_value):
    """Helper function to construct an interval object with a closed lower and open upper.

    For example:

    >>> closedopen(100.2, 800.9)
    [100.2, 800.9)
    """
    return Interval(Interval.CLOSED, lower_value, upper_value, Interval.OPEN)