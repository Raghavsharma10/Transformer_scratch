def some(predicate, *seqs):
    """
    >>> some(lambda x: x, [0, False, None])
    False
    >>> some(lambda x: x, [None, 0, 2, 3])
    2
    >>> some(operator.eq, [0,1,2], [2,1,0])
    True
    >>> some(operator.eq, [1,2], [2,1])
    False
    """
    try:
        if len(seqs) == 1: return ifilter(bool,imap(predicate, seqs[0])).next()
        else:             return ifilter(bool,starmap(predicate, izip(*seqs))).next()
    except StopIteration: return False