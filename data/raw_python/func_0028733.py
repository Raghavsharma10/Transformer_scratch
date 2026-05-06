def _pairwise(iterable):
    """
    itertools recipe
    "s -> (s0,s1), (s1,s2), (s2, s3), ...

    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)