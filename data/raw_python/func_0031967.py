def openclosed(lower_value, upper_value):
    """Helper function to construct an interval object with a open lower and closed upper.

    For example:

    >>> openclosed(100.2, 800.9)
    (100.2, 800.9]
    """
    return Interval(Interval.OPEN, lower_value, upper_value, Interval.CLOSED)