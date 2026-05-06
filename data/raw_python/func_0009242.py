def _distance_matrix_generic(x, centering, exponent=1):
    """Compute a centered distance matrix given a matrix."""
    _check_valid_dcov_exponent(exponent)

    x = _transform_to_2d(x)

    # Calculate distance matrices
    a = distances.pairwise_distances(x, exponent=exponent)

    # Double centering
    a = centering(a, out=a)

    return a