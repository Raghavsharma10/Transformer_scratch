def cv(data, units=False):
    """
    Calculate coefficient of variation (cv) of data. Mean and standard deviation
    are computed across time.


    Parameters
    ----------
    data : numpy.ndarray
        1st axis unit, 2nd axis time.
    units : bool
        Average `cv`.


    Returns
    -------
    numpy.ndarray
        If units=False, series of unit `cv`s.
    float
        If units=True, mean `cv` across units.


    Examples
    --------
    >>> cv(np.array([[1, 2, 3, 4, 5, 6], [11, 2, 3, 3, 4, 5]]))
    array([ 0.48795004,  0.63887656])

    >>> cv(np.array([[1, 2, 3, 4, 5, 6], [11, 2, 3, 3, 4, 5]]), units=True)
    0.56341330073710316

    """

    mu = mean(data, time=True)
    var = variance(data, time=True)
    cv = np.sqrt(var) / mu
    
    if units is True:
        return np.mean(cv)
    else:
        return cv