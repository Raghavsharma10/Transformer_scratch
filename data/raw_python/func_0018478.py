def distance(f1, f2):
    """\
    Distance between 2 features. The integer result is always positive or zero.
    If the features overlap or touch, it is zero.
    >>> from intersecter import Feature, distance
    >>> distance(Feature(1, 2), Feature(12, 13))
    10
    >>> distance(Feature(1, 2), Feature(2, 3))
    0
    >>> distance(Feature(1, 100), Feature(20, 30))
    0

    """
    if f1.end < f2.start: return f2.start - f1.end
    if f2.end < f1.start: return f1.start - f2.end
    return 0