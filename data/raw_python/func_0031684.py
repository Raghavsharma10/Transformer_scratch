def powerspec(data, tbin, Df=None, units=False, pointProcess=False):
    """
    Calculate (smoothed) power spectra of all timeseries in data.
    If units=True, power spectra are averaged across units.
    Note that averaging is done on power spectra rather than data.

    If pointProcess is True, power spectra are normalized by the length T of the
    time series.

 
    Parameters
    ----------
    data : numpy.ndarray
        1st axis unit, 2nd axis time.
    tbin : float
        Binsize in ms.
    Df : float/None, 
        Window width of sliding rectangular filter (smoothing),
        None is no smoothing.
    units : bool
        Average power spectrum.
    pointProcess : bool
        If set to True, powerspectrum is normalized to signal length T.


    Returns
    -------
    freq : tuple
        numpy.ndarray of frequencies.
    POW : tuple
        if units=False: 
            2 dim numpy.ndarray; 1st axis unit, 2nd axis frequency
        if units=True:  
            1 dim numpy.ndarray; frequency series

    
    Examples
    --------    
    >>> powerspec(np.array([analog_sig1, analog_sig2]), tbin, Df=Df)
    Out[1]: (freq,POW)
    >>> POW.shape
    Out[2]: (2,len(analog_sig1))

    >>> powerspec(np.array([analog_sig1, analog_sig2]), tbin, Df=Df, units=True)
    Out[1]: (freq,POW)
    >>> POW.shape
    Out[2]: (len(analog_sig1),)

    """

    freq, DATA = calculate_fft(data, tbin)
    df = freq[1] - freq[0]
    T = tbin * len(freq)
    POW = np.abs(DATA) ** 2
    if Df is not None:
        POW = [movav(x, Df, df) for x in POW]
        cut = int(Df / df)
        freq = freq[cut:]
        POW = np.array([x[cut:] for x in POW])
        POW = np.abs(POW)
    assert(len(freq) == len(POW[0]))
    if units is True:
        POW = mean(POW, units=units)
        assert(len(freq) == len(POW))
    if pointProcess:
        POW *= 1. / T * 1e3  # Normalization, power independent of T
    return freq, POW