def _u_distance_correlation_sqr_naive(x, y, exponent=1):
    """Bias-corrected distance correlation estimator between two matrices."""
    return _distance_sqr_stats_naive_generic(
        x, y,
        matrix_centered=_u_distance_matrix,
        product=u_product,
        exponent=exponent).correlation_xy