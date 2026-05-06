def atIndices(indexable, indices, default=__unique):
    r"""Return a list of items in `indexable` at positions `indices`.

    Examples:

    >>> atIndices([1,2,3], [1,1,0])
    [2, 2, 1]
    >>> atIndices([1,2,3], [1,1,0,4], 'default')
    [2, 2, 1, 'default']
    >>> atIndices({'a':3, 'b':0}, ['a'])
    [3]
    """
    if default is __unique:
        return [indexable[i] for i in indices]
    else:
        res = []
        for i in indices:
            try:
                res.append(indexable[i])
            except (IndexError, KeyError):
                res.append(default)
        return res