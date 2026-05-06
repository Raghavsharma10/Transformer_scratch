def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item