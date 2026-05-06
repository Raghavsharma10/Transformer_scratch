def splitAt(iterable, indices):
    r"""Yield chunks of `iterable`, split at the points in `indices`:

    >>> [l for l in splitAt(range(10), [2,5])]
    [[0, 1], [2, 3, 4], [5, 6, 7, 8, 9]]

    splits past the length of `iterable` are ignored:

    >>> [l for l in splitAt(range(10), [2,5,10])]
    [[0, 1], [2, 3, 4], [5, 6, 7, 8, 9]]


    """
    iterable = iter(iterable)
    now = 0
    for to in indices:
        try:
            res = []
            for i in range(now, to): res.append(iterable.next())
        except StopIteration: yield res; return
        yield res
        now = to
    res = list(iterable)
    if res: yield res