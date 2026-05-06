def param_squared_mean(ns_run, logw=None, simulate=False, param_ind=0):
    """Mean of the square of single parameter (second moment of its
    posterior distribution).

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
    param_ind: int, optional
        Index of parameter for which the second moment should be
        calculated. This corresponds to the column of ns_run['theta']
        which contains the parameter.

    Returns
    -------
    float
    """
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())  # protect against overflow
    w_relative /= np.sum(w_relative)
    return np.sum(w_relative * (ns_run['theta'][:, param_ind] ** 2))