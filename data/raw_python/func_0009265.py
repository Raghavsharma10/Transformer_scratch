def pairwise(function, x, y=None, **kwargs):
    """
    pairwise(function, x, y=None, *, pool=None, is_symmetric=None, **kwargs)

    Computes a dependency measure between each pair of elements.

    Parameters
    ----------
    function: Dependency measure function.
    x: iterable of array_like
        First list of random vectors. The columns of each vector correspond
        with the individual random variables while the rows are individual
        instances of the random vector.
    y: array_like
        Second list of random vectors. The columns of each vector correspond
        with the individual random variables while the rows are individual
        instances of the random vector. If None, the :math:`x` array is used.
    pool: object implementing multiprocessing.Pool interface
        Pool of processes/threads used to delegate computations.
    is_symmetric: bool or None
        If True, the dependency function is assumed to be symmetric. If False,
        it is assumed non-symmetric. If None (the default value), the attribute
        :code:`is_symmetric` of the function object is inspected to determine
        if the function is symmetric. If this attribute is absent, the function
        is assumed to not be symmetric.
    kwargs: dictionary
        Additional options necessary.

    Returns
    -------
    numpy ndarray
        A :math:`n \times m` matrix where the :math:`(i, j)`-th entry is the
        dependency between :math:`x[i]` and :math:`y[j]`.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = [np.array([[1, 1],
    ...                [2, 4],
    ...                [3, 8],
    ...                [4, 16]]),
    ...      np.array([[9, 10],
    ...                [11, 12],
    ...                [13, 14],
    ...                [15, 16]])
    ...     ]
    >>> b = [np.array([[0, 1],
    ...                [3, 1],
    ...                [6, 2],
    ...                [9, 3]]),
    ...      np.array([[5, 1],
    ...                [8, 1],
    ...                [13, 1],
    ...                [21, 1]])
    ...     ]
    >>> dcor.pairwise(dcor.distance_covariance, a)
    array([[4.61229635, 3.35991482],
           [3.35991482, 2.54950976]])
    >>> dcor.pairwise(dcor.distance_correlation, a, b)
    array([[0.98182263, 0.99901855],
           [0.99989466, 0.98320103]])

    A pool object can be used to improve performance for a large
    number of computations:

    >>> import multiprocessing
    >>> pool = multiprocessing.Pool()
    >>> dcor.pairwise(dcor.distance_correlation, a, b, pool=pool)
    array([[0.98182263, 0.99901855],
           [0.99989466, 0.98320103]])

    It is possible to force to consider that the function is symmetric or not
    (useful only if :math:`y` is :code:`None`):
    >>> dcor.pairwise(dcor.distance_covariance, a, is_symmetric=True)
    array([[4.61229635, 3.35991482],
           [3.35991482, 2.54950976]])

    >>> dcor.pairwise(dcor.distance_covariance, a, is_symmetric=False)
    array([[4.61229635, 3.35991482],
           [3.35991482, 2.54950976]])

    """

    return _pairwise_imp(function, x, y, **kwargs)