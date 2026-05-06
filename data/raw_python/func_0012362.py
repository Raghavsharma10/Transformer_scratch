def run_std_bootstrap(ns_run, estimator_list, **kwargs):
    """
    Uses bootstrap resampling to calculate an estimate of the
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
    kwargs: dict
        kwargs for run_bootstrap_values

    Returns
    -------
    output: 1d numpy array
        Sampling error on calculation result for each estimator in
        estimator_list.
    """
    bs_values = run_bootstrap_values(ns_run, estimator_list, **kwargs)
    stds = np.zeros(bs_values.shape[0])
    for j, _ in enumerate(stds):
        stds[j] = np.std(bs_values[j, :], ddof=1)
    return stds