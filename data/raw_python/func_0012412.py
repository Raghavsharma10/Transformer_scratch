def get_w_rel(ns_run, simulate=False):
    """Get the relative posterior weights of the samples, normalised so
    the maximum sample weight is 1. This is calculated from get_logw with
    protection against numerical overflows.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    simulate: bool, optional
        See the get_logw docstring for more details.

    Returns
    -------
    w_rel: 1d numpy array
        Relative posterior masses of points.
    """
    logw = get_logw(ns_run, simulate=simulate)
    return np.exp(logw - logw.max())