def include_after(predicate, num, iterative):
    """
    Return elements in `iterative` including `num`-after elements.

    >>> list(include_after(lambda x: x == 'b', 2, 'abcbcde'))
    ['b', 'c', 'b', 'c', 'd']

    """
    (it0, it1) = itertools.tee(iterative)
    ps = _forward_shifted_predicate(predicate, num, it1)
    return (e for (e, p) in zip(it0, ps) if p)