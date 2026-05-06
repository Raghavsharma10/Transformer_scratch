def partial_distance_covariance(x, y, z):
    """
    Partial distance covariance estimator.

    Compute the estimator for the partial distance covariance of the
    random vectors corresponding to :math:`x` and :math:`y` with respect
    to the random variable corresponding to :math:`z`.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    z: array_like
        Random vector with respect to which the partial distance covariance
        is computed. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.

    Returns
    -------
    numpy scalar
        Value of the estimator of the partial distance covariance.

    See Also
    --------
    partial_distance_correlation

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> c = np.array([[1, 3, 4],
    ...               [5, 7, 8],
    ...               [9, 11, 15],
    ...               [13, 15, 16]])
    >>> dcor.partial_distance_covariance(a, a, c) # doctest: +ELLIPSIS
    0.0024298...
    >>> dcor.partial_distance_covariance(a, b, c)
    0.0347030...
    >>> dcor.partial_distance_covariance(b, b, c)
    0.4956241...

    """
    a = _u_distance_matrix(x)
    b = _u_distance_matrix(y)
    c = _u_distance_matrix(z)

    proj = u_complementary_projection(c)

    return u_product(proj(a), proj(b))