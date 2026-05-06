def u_distance_covariance_sqr(x, y, **kwargs):
    """
    u_distance_covariance_sqr(x, y, *, exponent=1)

    Computes the unbiased estimator for the squared distance covariance
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
        Value of the unbiased estimator of the squared distance covariance.

    See Also
    --------
    distance_covariance
    distance_covariance_sqr

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
    >>> dcor.u_distance_covariance_sqr(a, a) # doctest: +ELLIPSIS
    42.6666666...
    >>> dcor.u_distance_covariance_sqr(a, b) # doctest: +ELLIPSIS
    -2.6666666...
    >>> dcor.u_distance_covariance_sqr(b, b) # doctest: +ELLIPSIS
    0.6666666...
    >>> dcor.u_distance_covariance_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
    -0.2996598...

    """
    if _can_use_fast_algorithm(x, y, **kwargs):
        return _u_distance_covariance_sqr_fast(x, y)
    else:
        return _u_distance_covariance_sqr_naive(x, y, **kwargs)