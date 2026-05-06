def include_before(predicate, num, iterative):
    """
    Return elements in `iterative` including `num`-before elements.

    >>> list(include_before(lambda x: x == 'd', 2, 'abcded'))
    ['b', 'c', 'd', 'e', 'd']

    """
    (it0, it1) = itertools.tee(iterative)
    ps = _backward_shifted_predicate(predicate, num, it1)
    return (e for (e, p) in zip(it0, ps) if p)