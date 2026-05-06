def _af_inv_scaled(x):
    """Scale a random vector for using the affinely invariant measures"""
    x = _transform_to_2d(x)

    cov_matrix = np.atleast_2d(np.cov(x, rowvar=False))

    cov_matrix_power = _mat_sqrt_inv(cov_matrix)

    return x.dot(cov_matrix_power)