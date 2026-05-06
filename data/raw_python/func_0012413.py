def get_logx(nlive, simulate=False):
    r"""Returns a logx vector showing the expected or simulated logx positions
    of points.

    The shrinkage factor between two points

    .. math:: t_i = X_{i-1} / X_{i}

    is distributed as the largest of :math:`n_i` uniform random variables
    between 1 and 0, where :math:`n_i` is the local number of live points.

    We are interested in

    .. math:: \log(t_i) = \log X_{i-1} - \log X_{i}

    which has expected value :math:`-1/n_i`.

    Parameters
    ----------
    nlive_array: 1d numpy array
        Ordered local number of live points present at each point's
        iso-likelihood contour.
    simulate: bool, optional
        Should log prior volumes logx be simulated from their distribution (if
        False their expected values are used).

    Returns
    -------
    logx: 1d numpy array
        log X values for points.
    """
    assert nlive.min() > 0, (
        'nlive contains zeros or negative values! nlive = ' + str(nlive))
    if simulate:
        logx_steps = np.log(np.random.random(nlive.shape)) / nlive
    else:
        logx_steps = -1 * (nlive.astype(float) ** -1)
    return np.cumsum(logx_steps)