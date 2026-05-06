def mean(data, units=False, time=False):
    """
    Function to compute mean of data


    Parameters
    ---------- 
    data : numpy.ndarray
        1st axis unit, 2nd axis time
    units : bool
        Average over units
    time : bool 
        Average over time


    Returns
    -------
    if units=False and time=False: 
        error
    if units=True: 
        1 dim numpy.ndarray; time series
    if time=True: 
        1 dim numpy.ndarray; series of unit means across time
    if units=True and time=True: 
        float; unit and time mean


    Examples
    --------
    >>> mean(np.array([[1, 2, 3], [4, 5, 6]]), units=True)
    array([ 2.5,  3.5,  4.5])

    >>> mean(np.array([[1, 2, 3], [4, 5, 6]]), time=True)
    array([ 2.,  5.])

    >>> mean(np.array([[1, 2, 3], [4, 5, 6]]), units=True,time=True)
    3.5

    """

    assert(units is not False or time is not False)
    if units is True and time is False:
        return np.mean(data, axis=0)
    elif units is False and time is True:
        return np.mean(data, axis=1)
    elif units is True and time is True:
        return np.mean(data)