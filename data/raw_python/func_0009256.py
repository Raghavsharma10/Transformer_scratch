def _distance_stats_sqr_fast_generic(x, y, dcov_function):
    """Compute the distance stats using the fast algorithm."""
    covariance_xy_sqr = dcov_function(x, y)
    variance_x_sqr = dcov_function(x, x)
    variance_y_sqr = dcov_function(y, y)
    denominator_sqr_signed = variance_x_sqr * variance_y_sqr
    denominator_sqr = np.absolute(denominator_sqr_signed)
    denominator = _sqrt(denominator_sqr)

    # Comparisons using a tolerance can change results if the
    # covariance has a similar order of magnitude
    if denominator == 0.0:
        correlation_xy_sqr = denominator.dtype.type(0)
    else:
        correlation_xy_sqr = covariance_xy_sqr / denominator

    return Stats(covariance_xy=covariance_xy_sqr,
                 correlation_xy=correlation_xy_sqr,
                 variance_x=variance_x_sqr,
                 variance_y=variance_y_sqr)