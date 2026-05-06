def run_ci_bootstrap(ns_run, estimator_list, **kwargs):
    """Uses bootstrap resampling to calculate credible intervals on the
    distribution of sampling errors (the uncertainty on the calculation)
    for a single nested sampling run.

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
    cred_int: float
    n_simulate: int
    ninit_sep: bool, optional

    Returns
    -------
    output: 1d numpy array
        Credible interval on sampling error on calculation result for each
        estimator in estimator_list.
    """
    cred_int = kwargs.pop('cred_int')   # No default, must specify
    bs_values = run_bootstrap_values(ns_run, estimator_list, **kwargs)
    # estimate specific confidence intervals
    # formulae for alpha CI on estimator T = 2 T(x) - G^{-1}(T(x*))
    # where G is the CDF of the bootstrap resamples
    expected_estimators = nestcheck.ns_run_utils.run_estimators(
        ns_run, estimator_list)
    cdf = ((np.asarray(range(bs_values.shape[1])) + 0.5) /
           bs_values.shape[1])
    ci_output = expected_estimators * 2
    for i, _ in enumerate(ci_output):
        ci_output[i] -= np.interp(
            1. - cred_int, cdf, np.sort(bs_values[i, :]))
    return ci_output