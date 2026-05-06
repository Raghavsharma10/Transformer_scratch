def weighted_quantile(probability, values, weights):
    """
    Get quantile estimate for input probability given weighted samples using
    linear interpolation.

    Parameters
    ----------
    probability: float
        Quantile to estimate - must be in open interval (0, 1).
        For example, use 0.5 for the median and 0.84 for the upper
        84% quantile.
    values: 1d numpy array
        Sample values.
    weights: 1d numpy array
        Corresponding sample weights (same shape as values).

    Returns
    -------
    quantile: float
    """
    assert 1 > probability > 0, (
        'credible interval prob= ' + str(probability) + ' not in (0, 1)')
    assert values.shape == weights.shape
    assert values.ndim == 1
    assert weights.ndim == 1
    sorted_inds = np.argsort(values)
    quantiles = np.cumsum(weights[sorted_inds]) - (0.5 * weights[sorted_inds])
    quantiles /= np.sum(weights)
    return np.interp(probability, quantiles, values[sorted_inds])