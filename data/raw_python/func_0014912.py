def notUnique(iterable, reportMax=INF):
    """Returns the elements in `iterable` that aren't unique; stops after it found
    `reportMax` non-unique elements.

    Examples:

    >>> list(notUnique([1,1,2,2,3,3]))
    [1, 2, 3]
    >>> list(notUnique([1,1,2,2,3,3], 1))
    [1]
    """
    hash = {}
    n=0
    if reportMax < 1:
        raise ValueError("`reportMax` must be >= 1 and is %r" % reportMax)
    for item in iterable:
        count = hash[item] = hash.get(item, 0) + 1
        if count > 1:
            yield item
            n += 1
            if n >= reportMax:
                return