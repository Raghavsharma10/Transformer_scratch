def compound_crossspec(a_data, tbin, Df=None, pointProcess=False):
    """
    Calculate cross spectra of compound signals.
    a_data is a list of datasets (a_data = [data1,data2,...]).
    For each dataset in a_data, the compound signal is calculated
    and the crossspectra between these compound signals is computed.
    
    If pointProcess=True, power spectra are normalized by the length T of the
    time series.


    Parameters
    ----------
    a_data : list of numpy.ndarrays
        Array: 1st axis unit, 2nd axis time.     
    tbin : float
        Binsize in ms.
    Df : float/None, 
        Window width of sliding rectangular filter (smoothing),
        None -> no smoothing.
    pointProcess : bool
        If set to True, crossspectrum is normalized to signal length `T`
                
    Returns
    -------
    freq : tuple
        numpy.ndarray of frequencies.
    CRO : tuple
        3 dim numpy.ndarray; 1st axis first compound signal, 2nd axis second
        compound signal, 3rd axis frequency.


    Examples
    --------
    >>> compound_crossspec([np.array([analog_sig1, analog_sig2]),
                            np.array([analog_sig3,analog_sig4])], tbin, Df=Df)
    Out[1]: (freq,CRO)
    >>> CRO.shape
    Out[2]: (2,2,len(analog_sig1))

    """

    a_mdata = []
    for data in a_data:
        a_mdata.append(np.sum(data, axis=0))  # calculate compound signals
    return crossspec(np.array(a_mdata), tbin, Df, units=False,
                     pointProcess=pointProcess)