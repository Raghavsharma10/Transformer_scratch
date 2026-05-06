def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l