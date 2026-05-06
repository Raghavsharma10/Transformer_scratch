def compound_powerspec(data, tbin, Df=None, pointProcess=False):
    """
    Calculate the power spectrum of the compound/sum signal.
    data is first summed across units, then the power spectrum is calculated.

    If pointProcess=True, power spectra are normalized by the length T of
    the time series.

    
    Parameters
    ----------
    data : numpy.ndarray, 
        1st axis unit, 2nd axis time
    tbin : float, 
        binsize in ms
    Df : float/None, 
        window width of sliding rectangular filter (smoothing),
        None -> no smoothing
    pointProcess : bool, 
        if set to True, powerspectrum is normalized to signal length T
                 


    Returns
    -------
    freq : tuple
        numpy.ndarray of frequencies
    POW : tuple
        1 dim numpy.ndarray, frequency series


    Examples
    --------
    >>> compound_powerspec(np.array([analog_sig1, analog_sig2]), tbin, Df=Df)
    Out[1]: (freq,POW)
    >>> POW.shape
    Out[2]: (len(analog_sig1),)

    """

    return powerspec([np.sum(data, axis=0)], tbin, Df=Df, units=True,
        pointProcess=pointProcess)