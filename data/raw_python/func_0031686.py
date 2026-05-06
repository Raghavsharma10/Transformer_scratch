def crossspec(data, tbin, Df=None, units=False, pointProcess=False):
    """
    Calculate (smoothed) cross spectra of data.
    If `units`=True, cross spectra are averaged across units.
    Note that averaging is done on cross spectra rather than data.

    Cross spectra are normalized by the length T of the time series -> no
    scaling with T.

    If pointProcess=True, power spectra are normalized by the length T of the
    time series.


    Parameters
    ----------
    data : numpy.ndarray, 
        1st axis unit, 2nd axis time
    tbin : float, 
        binsize in ms
    Df : float/None, 
        window width of sliding rectangular filter (smoothing),
        None -> no smoothing
    units : bool, 
        average cross spectrum
    pointProcess : bool, 
        if set to True, cross spectrum is normalized to signal length T

    
    Returns
    -------
    freq : tuple
        numpy.ndarray of frequencies
    CRO : tuple
        if `units`=True: 1 dim numpy.ndarray; frequency series
        if `units`=False:3 dim numpy.ndarray; 1st axis first unit,
            2nd axis second unit, 3rd axis frequency


    Examples
    --------    
    >>> crossspec(np.array([analog_sig1, analog_sig2]), tbin, Df=Df)
    Out[1]: (freq,CRO)
    >>> CRO.shape
    Out[2]: (2,2,len(analog_sig1))

    >>> crossspec(np.array([analog_sig1, analog_sig2]), tbin, Df=Df, units=True)
    Out[1]: (freq,CRO)
    >>> CRO.shape
    Out[2]: (len(analog_sig1),)

    """

    N = len(data)
    if units is True:
        # smoothing and normalization take place in powerspec
        # and compound_powerspec
        freq, POW = powerspec(data, tbin, Df=Df, units=True)
        freq_com, CPOW = compound_powerspec(data, tbin, Df=Df)
        assert(len(freq) == len(freq_com))
        assert(np.min(freq) == np.min(freq_com))
        assert(np.max(freq) == np.max(freq_com))
        CRO = 1. / (1. * N * (N - 1.)) * (CPOW - 1. * N * POW)
        assert(len(freq) == len(CRO))
    else:
        freq, DATA = calculate_fft(data, tbin)
        T = tbin * len(freq)
        df = freq[1] - freq[0]
        if Df is not None:
            cut = int(Df / df)
            freq = freq[cut:]
        CRO = np.zeros((N, N, len(freq)), dtype=complex)
        for i in range(N):
            for j in range(i + 1):
                tempij = DATA[i] * DATA[j].conj()
                if Df is not None:
                    tempij = movav(tempij, Df, df)[cut:]
                CRO[i, j] = tempij
                CRO[j, i] = CRO[i, j].conj()
        assert(len(freq) == len(CRO[0, 0]))
        if pointProcess:
            CRO *= 1. / T * 1e3  # normalization
    return freq, CRO