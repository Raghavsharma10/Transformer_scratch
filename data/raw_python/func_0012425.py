def r_mean(ns_run, logw=None, simulate=False):
    """Mean of the radial coordinate (magnitude of theta vector).

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

    Returns
    -------
    float
    """
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())
    r = np.sqrt(np.sum(ns_run['theta'] ** 2, axis=1))
    return np.sum(w_relative * r) / np.sum(w_relative)