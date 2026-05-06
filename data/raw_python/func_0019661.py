def function(data, maxt=None):
    """
    Calculate the autocorrelation function for a 1D time series.

    Parameters
    ----------
    data : numpy.ndarray (N,)
        The time series.

    Returns
    -------
    rho : numpy.ndarray (N,)
        An autocorrelation function.

    """
    data = np.atleast_1d(data)
    assert len(np.shape(data)) == 1, \
        "The autocorrelation function can only by computed " \
        + "on a 1D time series."
    if maxt is None:
        maxt = len(data)
    result = np.zeros(maxt, dtype=float)
    _acor.function(np.array(data, dtype=float), result)
    return result / result[0]