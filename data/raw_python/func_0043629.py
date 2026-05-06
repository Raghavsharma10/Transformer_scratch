def shape_factors(n, dim=2):
    """
    Returns a :obj:`numpy.ndarray` of factors :samp:`f` such
    that :samp:`(len(f) == {dim}) and (numpy.product(f) == {n})`.
    The returned factors are as *square* (*cubic*, etc) as possible.
    For example::

       >>> shape_factors(24, 1)
       array([24])
       >>> shape_factors(24, 2)
       array([4, 6])
       >>> shape_factors(24, 3)
       array([2, 3, 4])
       >>> shape_factors(24, 4)
       array([2, 2, 2, 3])
       >>> shape_factors(24, 5)
       array([1, 2, 2, 2, 3])
       >>> shape_factors(24, 6)
       array([1, 1, 2, 2, 2, 3])

    :type n: :obj:`int`
    :param n: Integer which is factored into :samp:`{dim}` factors.
    :type dim: :obj:`int`
    :param dim: Number of factors.
    :rtype: :obj:`numpy.ndarray`
    :return: A :samp:`({dim},)` shaped array of integers which are factors of :samp:`{n}`.
    """
    if dim <= 1:
        factors = [n, ]
    else:
        for f in range(int(n ** (1.0 / float(dim))) + 1, 0, -1):
            if (n % f) == 0:
                factors = [f, ] + list(shape_factors(n // f, dim=dim - 1))
                break

    factors.sort()
    return _np.array(factors)