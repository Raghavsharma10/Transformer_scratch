def per_window(sequence, n=1):
    """
    From http://stackoverflow.com/q/42220614/610569

        >>> list(per_window([1,2,3,4], n=2))
        [(1, 2), (2, 3), (3, 4)]
        >>> list(per_window([1,2,3,4], n=3))
        [(1, 2, 3), (2, 3, 4)]
    """
    start, stop = 0, n
    seq = list(sequence)
    while stop <= len(seq):
        yield tuple(seq[start:stop])
        start += 1
        stop += 1