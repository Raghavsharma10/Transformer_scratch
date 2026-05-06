def without(seq1, seq2):
    r"""Return a list with all elements in `seq2` removed from `seq1`, order
    preserved.

    Examples:

    >>> without([1,2,3,1,2], [1])
    [2, 3, 2]
    """
    if isSet(seq2): d2 = seq2
    else: d2 = set(seq2)
    return [elt for elt in seq1 if elt not in d2]