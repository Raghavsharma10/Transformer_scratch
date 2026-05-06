def beta_confidence_intervals(ci_X, ntrials, ci=0.95):
    """
    Compute confidence intervals of beta distributions.

    Parameters
    ----------
    ci_X : numpy.array
        Computed confidence interval estimate from `ntrials` experiments
    ntrials : int
        The number of trials that were run.
    ci : float, optional, default=0.95
        Confidence interval to report (e.g. 0.95 for 95% confidence interval)

    Returns
    -------
    Plow : float
        The lower bound of the symmetric confidence interval.
    Phigh : float
        The upper bound of the symmetric confidence interval.

    Examples
    --------

    >>> ci_X = np.random.rand(10,10)
    >>> ntrials = 100
    >>> [Plow, Phigh] = beta_confidence_intervals(ci_X, ntrials)

    """
    # Compute low and high confidence interval for symmetric CI about mean.
    ci_low = 0.5 - ci/2;
    ci_high = 0.5 + ci/2;

    # Compute for every element of ci_X.
    from scipy.stats import beta
    Plow = ci_X * 0.0;
    Phigh = ci_X * 0.0;
    for i in range(ci_X.shape[0]):
        for j in range(ci_X.shape[1]):
            Plow[i,j] = beta.ppf(ci_low, a = ci_X[i,j] * ntrials, b = (1-ci_X[i,j]) * ntrials);
            Phigh[i,j] = beta.ppf(ci_high, a = ci_X[i,j] * ntrials, b = (1-ci_X[i,j]) * ntrials);

    return [Plow, Phigh]