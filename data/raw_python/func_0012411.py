def get_logw(ns_run, simulate=False):
    r"""Calculates the log posterior weights of the samples (using logarithms
    to avoid overflow errors with very large or small values).

    Uses the trapezium rule such that the weight of point i is

    .. math:: w_i = \mathcal{L}_i (X_{i-1} - X_{i+1}) / 2

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    simulate: bool, optional
        Should log prior volumes logx be simulated from their distribution (if
        false their expected values are used).

    Returns
    -------
    logw: 1d numpy array
        Log posterior masses of points.
    """
    try:
        # find logX value for each point
        logx = get_logx(ns_run['nlive_array'], simulate=simulate)
        logw = np.zeros(ns_run['logl'].shape[0])
        # Vectorized trapezium rule: w_i prop to (X_{i-1} - X_{i+1}) / 2
        logw[1:-1] = log_subtract(logx[:-2], logx[2:]) - np.log(2)
        # Assign all prior volume closest to first point X_first to that point:
        # that is from logx=0 to logx=log((X_first + X_second) / 2)
        logw[0] = log_subtract(0, scipy.special.logsumexp([logx[0], logx[1]]) -
                               np.log(2))
        # Assign all prior volume closest to final point X_last to that point:
        # that is from logx=log((X_penultimate + X_last) / 2) to logx=-inf
        logw[-1] = scipy.special.logsumexp([logx[-2], logx[-1]]) - np.log(2)
        # multiply by likelihood (add in log space)
        logw += ns_run['logl']
        return logw
    except IndexError:
        if ns_run['logl'].shape[0] == 1:
            # If there is only one point in the run then assign all prior
            # volume X \in (0, 1) to that point, so the weight is just
            # 1 * logl_0 = logl_0
            return copy.deepcopy(ns_run['logl'])
        else:
            raise