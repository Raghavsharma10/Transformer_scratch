def _u_distance_covariance_sqr_naive(x, y, exponent=1):
    """
    Naive unbiased estimator for distance covariance.

    Computes the unbiased estimator for distance covariance between two
    matrices, using an :math:`O(N^2)` algorithm.
    """
    a = _u_distance_matrix(x, exponent=exponent)
    b = _u_distance_matrix(y, exponent=exponent)

    return u_product(a, b)