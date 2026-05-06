def _distance_sqr_stats_naive_generic(x, y, matrix_centered, product,
                                      exponent=1):
    """Compute generic squared stats."""
    a = matrix_centered(x, exponent=exponent)
    b = matrix_centered(y, exponent=exponent)

    covariance_xy_sqr = product(a, b)
    variance_x_sqr = product(a, a)
    variance_y_sqr = product(b, b)

    denominator_sqr = np.absolute(variance_x_sqr * variance_y_sqr)
    denominator = _sqrt(denominator_sqr)

    # Comparisons using a tolerance can change results if the
    # covariance has a similar order of magnitude
    if denominator == 0.0:
        correlation_xy_sqr = 0.0
    else:
        correlation_xy_sqr = covariance_xy_sqr / denominator

    return Stats(covariance_xy=covariance_xy_sqr,
                 correlation_xy=correlation_xy_sqr,
                 variance_x=variance_x_sqr,
                 variance_y=variance_y_sqr)