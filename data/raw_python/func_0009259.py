def distance_stats_sqr(x, y, **kwargs):
    """
    distance_stats_sqr(x, y, *, exponent=1)

    Computes the usual (biased) estimators for the squared distance covariance
    and squared distance correlation between two random vectors, and the
    individual squared distance variances.

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
        Squared distance covariance, squared distance correlation,
        squared distance variance of the first random vector and
        squared distance variance of the second random vector.

    See Also
    --------
    distance_covariance_sqr
    distance_correlation_sqr

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
    >>> dcor.distance_stats_sqr(a, a) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=52.0, correlation_xy=1.0, variance_x=52.0,
    variance_y=52.0)
    >>> dcor.distance_stats_sqr(a, b) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=1.0, correlation_xy=0.2773500...,
    variance_x=52.0, variance_y=0.25)
    >>> dcor.distance_stats_sqr(b, b) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=0.25, correlation_xy=1.0, variance_x=0.25,
    variance_y=0.25)
    >>> dcor.distance_stats_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
    ...                                 # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=0.3705904..., correlation_xy=0.4493308...,
    variance_x=2.7209220..., variance_y=0.25)

    """
    if _can_use_fast_algorithm(x, y, **kwargs):
        return _distance_stats_sqr_fast(x, y)
    else:
        return _distance_sqr_stats_naive_generic(
            x, y,
            matrix_centered=_distance_matrix,
            product=mean_product,
            **kwargs)