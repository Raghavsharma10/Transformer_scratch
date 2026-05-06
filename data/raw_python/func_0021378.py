def multinterp(x, y, xquery, slow=False):
    """Multiple linear interpolations

    Parameters
    ----------
    x : array_like, shape=(N,)
        sorted array of x values
    y : array_like, shape=(N, M)
        array of y values corresponding to each x value
    xquery : array_like, shape=(M,)
        array of query values
    slow : boolean, default=False
        if True, use slow method (used mainly for unit testing)

    Returns
    -------
    yquery : ndarray, shape=(M,)
        The interpolated values corresponding to each x query.
    """
    x, y, xquery = map(np.asarray, (x, y, xquery))
    assert x.ndim == 1
    assert xquery.ndim == 1
    assert y.shape == x.shape + xquery.shape

    # make sure xmin < xquery < xmax in all cases
    xquery = np.clip(xquery, x.min(), x.max())

    if slow:
        from scipy.interpolate import interp1d
        return np.array([interp1d(x, y)(xq) for xq, y in zip(xquery, y.T)])
    elif len(x) == 3:
        # Most common case: use a faster approach
        yq_lower = y[0] + (xquery - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
        yq_upper = y[1] + (xquery - x[1]) * (y[2] - y[1]) / (x[2] - x[1])
        return np.where(xquery < x[1], yq_lower, yq_upper)
    else:
        i = np.clip(np.searchsorted(x, xquery, side='right') - 1,
                    0, len(x) - 2)
        j = np.arange(len(xquery))
        return y[i, j] + ((xquery - x[i]) *
                          (y[i + 1, j] - y[i, j]) / (x[i + 1] - x[i]))