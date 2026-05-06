def _energy_distance_imp(x, y, exponent=1):
    """
    Real implementation of :func:`energy_distance`.

    This function is used to make parameter ``exponent`` keyword-only in
    Python 2.

    """
    x = _transform_to_2d(x)
    y = _transform_to_2d(y)

    _check_valid_energy_exponent(exponent)

    distance_xx = distances.pairwise_distances(x, exponent=exponent)
    distance_yy = distances.pairwise_distances(y, exponent=exponent)
    distance_xy = distances.pairwise_distances(x, y, exponent=exponent)

    return _energy_distance_from_distance_matrices(distance_xx=distance_xx,
                                                   distance_yy=distance_yy,
                                                   distance_xy=distance_xy)