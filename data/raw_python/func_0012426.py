def r_cred(ns_run, logw=None, simulate=False, probability=0.5):
    """One-tailed credible interval on the value of the radial coordinate
    (magnitude of theta vector).

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

    Returns
    -------
    float
    """
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())  # protect against overflow
    r = np.sqrt(np.sum(ns_run['theta'] ** 2, axis=1))
    return weighted_quantile(probability, r, w_relative)