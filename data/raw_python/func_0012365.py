def run_std_simulate(ns_run, estimator_list, n_simulate=None):
    """Uses the 'simulated weights' method to calculate an estimate of the
    standard deviation of the distribution of sampling errors (the
    uncertainty on the calculation) for a single nested sampling run.

    Note that the simulated weights method is not accurate for parameter
    estimation calculations.

    For more details about the simulated weights method for estimating sampling
    errors see 'Sampling errors in nested sampling parameter estimation'
    (Higson et al. 2018).

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
    estimator_list: list of functions for estimating quantities (such as the
        bayesian evidence or mean of parameters) from nested sampling runs.
        Example functions can be found in estimators.py. Each should have
        arguments: func(ns_run, logw=None)
    n_simulate: int

    Returns
    -------
    output: 1d numpy array
        Sampling error on calculation result for each estimator in
        estimator_list.
    """
    assert n_simulate is not None, 'run_std_simulate: must specify n_simulate'
    all_values = np.zeros((len(estimator_list), n_simulate))
    for i in range(n_simulate):
        all_values[:, i] = nestcheck.ns_run_utils.run_estimators(
            ns_run, estimator_list, simulate=True)
    stds = np.zeros(all_values.shape[0])
    for i, _ in enumerate(stds):
        stds[i] = np.std(all_values[i, :], ddof=1)
    return stds