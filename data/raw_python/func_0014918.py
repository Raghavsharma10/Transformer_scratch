def iterate(f, n=None, last=__unique):
    """
    >>> list(iterate(lambda x:x//2)(128))
    [128, 64, 32, 16, 8, 4, 2, 1, 0]
    >>> list(iterate(lambda x:x//2, n=2)(128))
    [128, 64]
    """
    if n is not None:
        def funciter(start):
            for i in xrange(n): yield start; start = f(start)
    else:
        def funciter(start):
            while True:
                yield start
                last = f(start)
                if last == start: return
                last, start = start, last
    return funciter