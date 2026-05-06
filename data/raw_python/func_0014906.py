def bipart(func, seq):
    r"""Like a partitioning version of `filter`. Returns
    ``[itemsForWhichFuncReturnedFalse, itemsForWhichFuncReturnedTrue]``.

    Example:

    >>> bipart(bool, [1,None,2,3,0,[],[0]])
    [[None, 0, []], [1, 2, 3, [0]]]
    """

    if func is None: func = bool
    res = [[],[]]
    for i in seq: res[not not func(i)].append(i)
    return res