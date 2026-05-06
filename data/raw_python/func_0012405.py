def run_estimators(ns_run, estimator_list, simulate=False):
    """Calculates values of list of quantities (such as the Bayesian evidence
    or mean of parameters) for a single nested sampling run.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    estimator_list: list of functions for estimating quantities from nested
        sampling runs. Example functions can be found in estimators.py. Each
        should have arguments: func(ns_run, logw=None).
    simulate: bool, optional
        See get_logw docstring.

    Returns
    -------
    output: 1d numpy array
        Calculation result for each estimator in estimator_list.
    """
    logw = get_logw(ns_run, simulate=simulate)
    output = np.zeros(len(estimator_list))
    for i, est in enumerate(estimator_list):
        output[i] = est(ns_run, logw=logw)
    return output