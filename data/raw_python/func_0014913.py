def unweave(iterable, n=2):
    r"""Divide `iterable` in `n` lists, so that every `n`th element belongs to
    list `n`.

    Example:

    >>> unweave((1,2,3,4,5), 3)
    [[1, 4], [2, 5], [3]]
    """
    res = [[] for i in range(n)]
    i = 0
    for x in iterable:
        res[i % n].append(x)
        i += 1
    return res