def dropwhilenot(func, iterable):
    """
    >>> list(dropwhilenot(lambda x:x==3, range(10)))
    [3, 4, 5, 6, 7, 8, 9]
    """
    iterable = iter(iterable)
    for x in iterable:
        if func(x): break
    else: return
    yield x
    for x in iterable:
        yield x