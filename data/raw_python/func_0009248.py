def _distance_covariance_sqr_naive(x, y, exponent=1):
    """
    Naive biased estimator for distance covariance.

    Computes the unbiased estimator for distance covariance between two
    matrices, using an :math:`O(N^2)` algorithm.
    """
    a = _distance_matrix(x, exponent=exponent)
    b = _distance_matrix(y, exponent=exponent)

    return mean_product(a, b)