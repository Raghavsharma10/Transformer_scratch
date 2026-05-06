def union(seq1=(), *seqs):
    r"""Return the set union of `seq1` and `seqs`, duplicates removed, order random.

    Examples:
    >>> union()
    []
    >>> union([1,2,3])
    [1, 2, 3]
    >>> union([1,2,3], {1:2, 5:1})
    [1, 2, 3, 5]
    >>> union((1,2,3), ['a'], "bcd")
    ['a', 1, 2, 3, 'd', 'b', 'c']
    >>> union([1,2,3], iter([0,1,1,1]))
    [0, 1, 2, 3]

    """
    if not seqs: return list(seq1)
    res = set(seq1)
    for seq in seqs:
        res.update(set(seq))
    return list(res)