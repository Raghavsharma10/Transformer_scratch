def DiscreteGamma(alpha, beta, ncats):
    """Returns category means for discretized gamma distribution.

    The distribution is evenly divided into categories, and the
    mean of each category is returned.

    Args:
        `alpha` (`float` > 0)
            Shape parameter of gamma distribution.
        `beta` (`float` > 0)
            Inverse scale parameter of gamma distribution.
        `ncats` (`int` > 1)
            Number of categories in discretized gamma distribution.

    Returns:
        `catmeans` (`scipy.ndarray` of `float`, shape `(ncats,)`)
            `catmeans[k]` is the mean of category `k` where
            `0 <= k < ncats`.

    Check that we get values in Figure 1 of Yang, J Mol Evol, 39:306-314
    >>> catmeans = DiscreteGamma(0.5, 0.5, 4)
    >>> scipy.allclose(catmeans, scipy.array([0.0334, 0.2519, 0.8203, 2.8944]), atol=1e-4)
    True

    Make sure we get expected mean of alpha / beta
    >>> alpha = 0.6
    >>> beta = 0.8
    >>> ncats = 6
    >>> catmeans = DiscreteGamma(alpha, beta, ncats)
    >>> scipy.allclose(catmeans.sum() / ncats, alpha / beta)
    True
    """
    alpha = float(alpha)
    beta = float(beta)
    assert alpha > 0
    assert beta > 0
    assert ncats > 1
    scale = 1.0 / beta
    catmeans = scipy.ndarray(ncats, dtype='float')
    for k in range(ncats):
        if k == 0:
            lower = 0.0
            gammainc_lower = 0.0
        else:
            lower = upper
            gammainc_lower = gammainc_upper
        if k == ncats - 1:
            upper = float('inf')
            gammainc_upper = 1.0
        else:
            upper = scipy.stats.gamma.ppf((k + 1) / float(ncats), alpha,
                    scale=scale)
            gammainc_upper = scipy.special.gammainc(alpha + 1, upper * beta)
        catmeans[k] = alpha * ncats * (gammainc_upper - gammainc_lower) / beta
    assert scipy.allclose(catmeans.sum() / ncats, alpha / beta), (
            "catmeans is {0}, mean of catmeans is {1}, expected mean "
            "alpha / beta = {2} / {3} = {4}").format(catmeans,
            catmeans.sum() / ncats, alpha, beta, alpha / beta)
    return catmeans