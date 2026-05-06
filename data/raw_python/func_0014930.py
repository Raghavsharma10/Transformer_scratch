def every(predicate, *iterables):
    r"""Like `some`, but only returns `True` if all the elements of `iterables`
    satisfy `predicate`.

    Examples:
    >>> every(bool, [])
    True
    >>> every(bool, [0])
    False
    >>> every(bool, [1,1])
    True
    >>> every(operator.eq, [1,2,3],[1,2])
    True
    >>> every(operator.eq, [1,2,3],[0,2])
    False
    """
    try:
        if len(iterables) == 1: ifilterfalse(predicate, iterables[0]).next()
        else:                  ifilterfalse(bool, starmap(predicate, izip(*iterables))).next()
    except StopIteration: return True
    else: return False