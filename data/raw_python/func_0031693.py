def fano(data, units=False):
    """
    Calculate fano factor (FF) of data. Mean and variance are computed across
    time.


    Parameters
    ----------   
    data : numpy.ndarray 
        1st axis unit, 2nd axis time.
    units : bool
        Average `FF`.


    Returns
    -------
    numpy.ndarray
        If units=False, series of unit FFs.
    float
        If units=True, mean FF across units.


    Examples
    --------
    >>> fano(np.array([[1, 2, 3, 4, 5, 6], [11, 2, 3, 3, 4, 5]]))
    array([ 0.83333333,  1.9047619 ])

    >>> fano(np.array([[1, 2, 3, 4, 5, 6], [11, 2, 3, 3, 4, 5]]), units=True)
    1.3690476190476191

    """
    mu = mean(data, time=True)
    var = variance(data, time=True)
    ff = var / mu
    
    if units is True:
        return np.mean(ff)
    else:
        return ff