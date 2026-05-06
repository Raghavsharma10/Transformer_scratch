def param_mean(ns_run, logw=None, simulate=False, param_ind=0,
               handle_indexerror=False):
    """Mean of a single parameter (single component of theta).

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
        Index of parameter for which the mean should be calculated. This
        corresponds to the column of ns_run['theta'] which contains the
        parameter.
    handle_indexerror: bool, optional
        Make the function function return nan rather than raising an
        IndexError if param_ind >= ndim. This is useful when applying
        the same list of estimators to data sets of different dimensions.

    Returns
    -------
    float
    """
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())
    try:
        return (np.sum(w_relative * ns_run['theta'][:, param_ind])
                / np.sum(w_relative))
    except IndexError:
        if handle_indexerror:
            return np.nan
        else:
            raise