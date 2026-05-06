def distance_stats(x, y, **kwargs):
    """
    distance_stats(x, y, *, exponent=1)

    Computes the usual (biased) estimators for the distance covariance
    and distance correlation between two random vectors, and the
    individual distance variances.

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
    Stats
        Distance covariance, distance correlation,
        distance variance of the first random vector and
        distance variance of the second random vector.

    See Also
    --------
    distance_covariance
    distance_correlation

    Notes
    -----
    It is less efficient to compute the statistics separately, rather than
    using this function, because some computations can be shared.

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
    >>> dcor.distance_stats(a, a) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=7.2111025..., correlation_xy=1.0,
    variance_x=7.2111025..., variance_y=7.2111025...)
    >>> dcor.distance_stats(a, b) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=1.0, correlation_xy=0.5266403...,
    variance_x=7.2111025..., variance_y=0.5)
    >>> dcor.distance_stats(b, b) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=0.5, correlation_xy=1.0, variance_x=0.5,
    variance_y=0.5)
    >>> dcor.distance_stats(a, b, exponent=0.5) # doctest: +ELLIPSIS
    ...                             # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=0.6087614..., correlation_xy=0.6703214...,
    variance_x=1.6495217..., variance_y=0.5)

    """
    return Stats(*[_sqrt(s) for s in distance_stats_sqr(x, y, **kwargs)])