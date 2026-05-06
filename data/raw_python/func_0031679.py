def centralize(data, time=False, units=False):
    """
    Function to subtract the mean across time and/or across units from data
    
    
    Parameters
    ----------  
    data : numpy.ndarray
        1D or 2D array containing time series, 1st index: unit, 2nd index: time
    time : bool
        True: subtract mean across time.
    units : bool
        True: subtract mean across units.
            
    
    Returns
    -------
    numpy.ndarray
        1D or 0D array of centralized signal.
        
    """
    assert(time is not False or units is not False)
    res = copy.copy(data)
    
    if time is True:
        res = np.array([x - np.mean(x) for x in res])
    
    if units is True:
        res = np.array(res - np.mean(res, axis=0))
    
    return res