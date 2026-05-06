def run_bootstrap_values(ns_run, estimator_list, **kwargs):
    """Uses bootstrap resampling to calculate an estimate of the
    standard deviation of the distribution of sampling errors (the
    uncertainty on the calculation) for a single nested sampling run.

    For more details about bootstrap resampling for estimating sampling
    errors see 'Sampling errors in nested sampling parameter estimation'
    (Higson et al. 2018).

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
    estimator_list: list of functions for estimating quantities (such as the
        Bayesian evidence or mean of parameters) from nested sampling runs.
        Example functions can be found in estimators.py. Each should have
        arguments: func(ns_run, logw=None)
    n_simulate: int
    ninit_sep: bool, optional
        For dynamic runs: resample initial threads and dynamically added
        threads separately. Useful when there are only a few threads which
        start by sampling the whole prior, as errors occur if none of these are
        included in the bootstrap resample.
    flip_skew: bool, optional
        Determine if distribution of bootstrap values should be flipped about
        its mean to better represent our probability distribution on the true
        value - see "Bayesian astrostatistics: a backward look to the future"
        (Loredo, 2012 Figure 2) for an explanation.
        If true, the samples :math:`X` are mapped to :math:`2 \mu - X`, where
        :math:`\mu` is the mean sample value.
        This leaves the mean and standard deviation unchanged.
    random_seeds: list, optional
        list of random_seed arguments for bootstrap_resample_run.
        Defaults to range(n_simulate) in order to give reproducible results.

    Returns
    -------
    output: 1d numpy array
        Sampling error on calculation result for each estimator in
        estimator_list.
    """
    ninit_sep = kwargs.pop('ninit_sep', False)
    flip_skew = kwargs.pop('flip_skew', True)
    n_simulate = kwargs.pop('n_simulate')  # No default, must specify
    random_seeds = kwargs.pop('random_seeds', range(n_simulate))
    assert len(random_seeds) == n_simulate
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    threads = nestcheck.ns_run_utils.get_run_threads(ns_run)
    bs_values = np.zeros((len(estimator_list), n_simulate))
    for i, random_seed in enumerate(random_seeds):
        ns_run_temp = bootstrap_resample_run(
            ns_run, threads=threads, ninit_sep=ninit_sep,
            random_seed=random_seed)
        bs_values[:, i] = nestcheck.ns_run_utils.run_estimators(
            ns_run_temp, estimator_list)
        del ns_run_temp
    if flip_skew:
        estimator_means = np.mean(bs_values, axis=1)
        for i, mu in enumerate(estimator_means):
            bs_values[i, :] = (2 * mu) - bs_values[i, :]
    return bs_values