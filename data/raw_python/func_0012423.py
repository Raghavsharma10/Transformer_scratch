def param_cred(ns_run, logw=None, simulate=False, probability=0.5,
               param_ind=0):
    """One-tailed credible interval on the value of a single parameter
    (component of theta).

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see the data_processing module
        docstring for more details).
    logw: None or 1d numpy array, optional
        Log weights of samples.
    simulate: bool, optional
        Passed to ns_run_utils.get_logw if logw needs to be
        calculated.
    probability: float, optional
        Quantile to estimate - must be in open interval (0, 1).
        For example, use 0.5 for the median and 0.84 for the upper
        84% quantile. Passed to weighted_quantile.
    param_ind: int, optional
        Index of parameter for which the credible interval should be
        calculated. This corresponds to the column of ns_run['theta']
        which contains the parameter.

    Returns
    -------
    float
    """
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())  # protect against overflow
    return weighted_quantile(probability, ns_run['theta'][:, param_ind],
                             w_relative)