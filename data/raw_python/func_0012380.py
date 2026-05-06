def weighted_1d_gaussian_kde(x, samples, weights):
    """Gaussian kde with weighted samples (1d only). Uses Scott bandwidth
    factor.

    When all the sample weights are equal, this is equivalent to

    kde = scipy.stats.gaussian_kde(theta)
    return kde(x)

    When the weights are not all equal, we compute the effective number
    of samples as the information content (Shannon entropy)

    nsamp_eff = exp(- sum_i (w_i log(w_i)))

    Alternative ways to estimate nsamp_eff include Kish's formula

    nsamp_eff = (sum_i w_i) ** 2 / (sum_i w_i ** 2)

    See https://en.wikipedia.org/wiki/Effective_sample_size and "Effective
    sample size for importance sampling based on discrepancy measures"
    (Martino et al. 2017) for more information.

    Parameters
    ----------
    x: 1d numpy array
        Coordinates at which to evaluate the kde.
    samples: 1d numpy array
        Samples from which to calculate kde.
    weights: 1d numpy array of same shape as samples
        Weights of each point. Need not be normalised as this is done inside
        the function.

    Returns
    -------
    result: 1d numpy array of same shape as x
        Kde evaluated at x values.
    """
    assert x.ndim == 1
    assert samples.ndim == 1
    assert samples.shape == weights.shape
    # normalise weights and find effective number of samples
    weights /= np.sum(weights)
    nz_weights = weights[np.nonzero(weights)]
    nsamp_eff = np.exp(-1. * np.sum(nz_weights * np.log(nz_weights)))
    # Calculate the weighted sample variance
    mu = np.sum(weights * samples)
    var = np.sum(weights * ((samples - mu) ** 2))
    var *= nsamp_eff / (nsamp_eff - 1)  # correct for bias using nsamp_eff
    # Calculate bandwidth
    scott_factor = np.power(nsamp_eff, -1. / (5))  # 1d Scott factor
    sig = np.sqrt(var) * scott_factor
    # Calculate and weight residuals
    xx, ss = np.meshgrid(x, samples)
    chisquared = ((xx - ss) / sig) ** 2
    energy = np.exp(-0.5 * chisquared) / np.sqrt(2 * np.pi * (sig ** 2))
    result = np.sum(energy * weights[:, np.newaxis], axis=0)
    return result