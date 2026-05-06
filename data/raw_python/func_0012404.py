def array_ratio_std(values_n, sigmas_n, values_d, sigmas_d):
    r"""Gives error on the ratio of 2 floats or 2 1-dimensional arrays given
    their values and uncertainties. This assumes the covariance = 0, and that
    the input uncertainties are small compared to the corresponding input
    values. _n and _d denote the numerator and denominator respectively.

    Parameters
    ----------
    values_n: float or numpy array
        Numerator values.
    sigmas_n: float or numpy array
        :math:`1\sigma` uncertainties on values_n.
    values_d: float or numpy array
        Denominator values.
    sigmas_d: float or numpy array
        :math:`1\sigma` uncertainties on values_d.

    Returns
    -------
    std: float or numpy array
        :math:`1\sigma` uncertainty on values_n / values_d.
    """
    std = np.sqrt((sigmas_n / values_n) ** 2 + (sigmas_d / values_d) ** 2)
    std *= (values_n / values_d)
    return std