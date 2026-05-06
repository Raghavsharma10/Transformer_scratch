def drop(n, it, constructor=list):
    """
    >>> first(10,drop(10,xrange(sys.maxint),iter))
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    """
    return constructor(itertools.islice(it,n,None))