def _distance_correlation_sqr_naive(x, y, exponent=1):
    """Biased distance correlation estimator between two matrices."""
    return _distance_sqr_stats_naive_generic(
        x, y,
        matrix_centered=_distance_matrix,
        product=mean_product,
        exponent=exponent).correlation_xy