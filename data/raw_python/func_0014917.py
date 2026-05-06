def group(iterable, n=2, pad=__unique):
    r"""Iterate `n`-wise (default pairwise)  over `iter`.
    Examples:

    >>> for (first, last) in group("Akira Kurosawa John Ford".split()):
    ...     print "given name: %s surname: %s" % (first, last)
    ...
    given name: Akira surname: Kurosawa
    given name: John surname: Ford
    >>>
    >>> # both contain the same number of pairs
    >>> list(group(range(9))) == list(group(range(8)))
    True
    >>> # with n=3
    >>> list(group(range(10), 3))
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    >>> list(group(range(10), 3, pad=0))
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 0, 0)]
    """
    assert n>0    # ensure it doesn't loop forever
    if pad is not __unique: it = chain(iterable, (pad,)*(n-1))
    else:                   it = iter(iterable)
    perTuple = xrange(n)
    while True:
        yield tuple([it.next() for i in perTuple])