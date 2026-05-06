def nlargest(n, mapping):
    """
    Takes a mapping and returns the n keys associated with the largest values
    in descending order. If the mapping has fewer than n items, all its keys
    are returned.

    Equivalent to:
        ``next(zip(*heapq.nlargest(mapping.items(), key=lambda x: x[1])))``

    Returns
    -------
    list of up to n keys from the mapping

    """
    try:
        it = mapping.iteritems()
    except AttributeError:
        it = iter(mapping.items())
    pq = minpq()
    try:
        for i in range(n):
            pq.additem(*next(it))
    except StopIteration:
        pass
    try:
        while it:
            pq.pushpopitem(*next(it))
    except StopIteration:
        pass
    out = list(pq.popkeys())
    out.reverse()
    return out