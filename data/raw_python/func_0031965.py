def open(lower_value, upper_value):
    """Helper function to construct an interval object with open lower and upper.

    For example:

    >>> open(100.2, 800.9)
    (100.2, 800.9)
    """
    return Interval(Interval.OPEN, lower_value, upper_value, Interval.OPEN)