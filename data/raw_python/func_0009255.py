def _distance_covariance_sqr_fast_generic(
        x, y, unbiased=False):  # pylint:disable=too-many-locals
    # This function has many locals so it can be compared
    # with the original algorithm.
    """Fast algorithm for the squared distance covariance."""
    x = np.asarray(x)
    y = np.asarray(y)

    x = np.ravel(x)
    y = np.ravel(y)

    n = x.shape[0]
    assert n > 3
    assert n == y.shape[0]
    temp = range(n)

    # Step 1
    ix0 = np.argsort(x)
    vx = x[ix0]

    ix = np.zeros(n, dtype=int)
    ix[ix0] = temp

    iy0 = np.argsort(y)
    vy = y[iy0]

    iy = np.zeros(n, dtype=int)
    iy[iy0] = temp

    # Step 2
    sx = np.cumsum(vx)
    sy = np.cumsum(vy)

    # Step 3
    alpha_x = ix
    alpha_y = iy

    beta_x = sx[ix] - vx[ix]
    beta_y = sy[iy] - vy[iy]

    # Step 4
    x_dot = np.sum(x)
    y_dot = np.sum(y)

    # Step 5
    a_i_dot = x_dot + (2 * alpha_x - n) * x - 2 * beta_x
    b_i_dot = y_dot + (2 * alpha_y - n) * y - 2 * beta_y

    sum_ab = np.sum(a_i_dot * b_i_dot)

    # Step 6
    a_dot_dot = 2 * np.sum(alpha_x * x) - 2 * np.sum(beta_x)
    b_dot_dot = 2 * np.sum(alpha_y * y) - 2 * np.sum(beta_y)

    # Step 7
    gamma_1 = _partial_sum_2d(x, y, np.ones(n, dtype=x.dtype))
    gamma_x = _partial_sum_2d(x, y, x)
    gamma_y = _partial_sum_2d(x, y, y)
    gamma_xy = _partial_sum_2d(x, y, x * y)

    # Step 8
    aijbij = np.sum(x * y * gamma_1 + gamma_xy - x * gamma_y - y * gamma_x)

    if unbiased:
        d3 = (n - 3)
        d2 = (n - 2)
        d1 = (n - 1)
    else:
        d3 = d2 = d1 = n

    # Step 9
    d_cov = (aijbij / n / d3 - 2 * sum_ab / n / d2 / d3 +
             a_dot_dot / n * b_dot_dot / d1 / d2 / d3)

    return d_cov