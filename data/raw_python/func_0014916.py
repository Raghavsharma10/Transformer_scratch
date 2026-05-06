def window(iterable, n=2, s=1):
    r"""Move an `n`-item (default 2) windows `s` steps (default 1) at a time
    over `iterable`.

    Examples:

    >>> list(window(range(6),2))
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    >>> list(window(range(6),3))
    [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]
    >>> list(window(range(6),3, 2))
    [(0, 1, 2), (2, 3, 4)]
    >>> list(window(range(5),3,2)) == list(window(range(6),3,2))
    True
    """
    assert n >= s
    last = []
    for elt in iterable:
        last.append(elt)
        if len(last) == n: yield tuple(last); last=last[s:]