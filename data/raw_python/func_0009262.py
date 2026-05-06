def distance_correlation_sqr(x, y, **kwargs):
    """
    distance_correlation_sqr(x, y, *, exponent=1)

    Computes the usual (biased) estimator for the squared distance correlation
    between two random vectors.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
        Equivalently, it is twice the Hurst parameter of fractional Brownian
        motion.

    Returns
    -------
    numpy scalar
        Value of the biased estimator of the squared distance correlation.

    See Also
    --------
    distance_correlation
    u_distance_correlation_sqr

    Notes
    -----
    The algorithm uses the fast distance covariance algorithm proposed in
    :cite:`b-fast_distance_correlation` when possible.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_correlation_sqr(a, a)
    1.0
    >>> dcor.distance_correlation_sqr(a, b) # doctest: +ELLIPSIS
    0.2773500...
    >>> dcor.distance_correlation_sqr(b, b)
    1.0
    >>> dcor.distance_correlation_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
    0.4493308...

    """
    if _can_use_fast_algorithm(x, y, **kwargs):
        return _distance_correlation_sqr_fast(x, y)
    else:
        return _distance_correlation_sqr_naive(x, y, **kwargs)