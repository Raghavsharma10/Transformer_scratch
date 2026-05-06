def distance_correlation_af_inv_sqr(x, y):
    """
    Square of the affinely invariant distance correlation.

    Computes the estimator for the square of the affinely invariant distance
    correlation between two random vectors.

    .. warning:: The return value of this function is undefined when the
                 covariance matrix of :math:`x` or :math:`y` is singular.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.

    Returns
    -------
    numpy scalar
        Value of the estimator of the squared affinely invariant
        distance correlation.

    See Also
    --------
    distance_correlation
    u_distance_correlation

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 3, 2, 5],
    ...               [5, 7, 6, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 15, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_correlation_af_inv_sqr(a, a)
    1.0
    >>> dcor.distance_correlation_af_inv_sqr(a, b) # doctest: +ELLIPSIS
    0.5773502...
    >>> dcor.distance_correlation_af_inv_sqr(b, b)
    1.0

    """
    x = _af_inv_scaled(x)
    y = _af_inv_scaled(y)

    correlation = distance_correlation_sqr(x, y)
    return 0 if np.isnan(correlation) else correlation