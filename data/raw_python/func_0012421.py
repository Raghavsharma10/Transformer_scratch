def evidence(ns_run, logw=None, simulate=False):
    r"""Bayesian evidence :math:`\log \mathcal{Z}`.

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
    return np.exp(scipy.special.logsumexp(logw))