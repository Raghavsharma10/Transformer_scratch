def flatten(seq, isSeq=isSeq):
    r"""Returns a flattened version of a sequence `seq` as a `list`.
    Parameters:

     - `seq`: The sequence to be flattened (any iterable).
     - `isSeq`: The function called to determine whether something is a
        sequence (default: `isSeq`). *Beware that this function should
        **never** test positive for strings, because they are no real
        sequences and thus cause infinite recursion.*

    Examples:

    >>> flatten([1,[2,3,(4,[5,6]),7,8]])
    [1, 2, 3, 4, 5, 6, 7, 8]
    >>> # flaten only lists
    >>> flatten([1,[2,3,(4,[5,6]),7,8]], isSeq=lambda x:isinstance(x, list))
    [1, 2, 3, (4, [5, 6]), 7, 8]
    >>> flatten([1,2])
    [1, 2]
    >>> flatten([])
    []
    >>> flatten('123')
    ['1', '2', '3']
    """
    return [a for elt in seq
            for a in (isSeq(elt) and flatten(elt, isSeq) or
                      [elt])]