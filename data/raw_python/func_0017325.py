def asarray(x, dtype=None):
    '''Convert ``x`` into a ``numpy.ndarray``.'''
    iterable = scalarasiter(x)
    if isinstance(iterable, ndarray):
        return iterable
    else:
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)
        if dtype == object_type:
            a = ndarray((len(iterable),), dtype=dtype)
            for i,v in enumerate(iterable):
                a[i] = v
            return a
        else:
            return array(iterable, dtype=dtype)