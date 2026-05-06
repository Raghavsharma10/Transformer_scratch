def groupby_count(i, key=None, force_keys=None):
    """ Aggregate iterator values into buckets based on how frequently the
    values appear.

    Example::

        >>> list(groupby_count([1, 1, 1, 2, 3]))
        [(1, 3), (2, 1), (3, 1)]
    """
    counter = defaultdict(lambda: 0)
    if not key:
        key = lambda o: o

    for k in i:
        counter[key(k)] += 1

    if force_keys:
        for k in force_keys:
            counter[k] += 0

    return counter.items()