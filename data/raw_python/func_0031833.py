def include_context(predicate, num, iterative):
    """
    Return elements in `iterative` including `num` before and after elements.

    >>> ''.join(include_context(lambda x: x == '!', 2, 'bb!aa__bb!aa'))
    'bb!aabb!aa'

    """
    (it0, it1, it2) = itertools.tee(iterative, 3)
    psf = _forward_shifted_predicate(predicate, num, it1)
    psb = _backward_shifted_predicate(predicate, num, it2)
    return (e for (e, pf, pb) in zip(it0, psf, psb) if pf or pb)