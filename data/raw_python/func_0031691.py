def coherence(freq, power, cross):
    """
    Calculate frequency resolved coherence for given power- and crossspectra.


    Parameters
    ----------
    freq : numpy.ndarray
        Frequencies, 1 dim array.
    power : numpy.ndarray
        Power spectra, 1st axis units, 2nd axis frequencies.
    cross : numpy.ndarray, 
        Cross spectra, 1st axis units, 2nd axis units, 3rd axis frequencies.


    Returns
    -------
    freq: tuple
        1 dim numpy.ndarray of frequencies.
    coh: tuple
        ndim 3 numpy.ndarray of coherences, 1st axis units, 2nd axis units,
        3rd axis frequencies.

    """

    N = len(power)
    coh = np.zeros(np.shape(cross))
    
    for i in range(N):
        for j in range(N):
            coh[i, j] = cross[i, j] / np.sqrt(power[i] * power[j])
    
    assert(len(freq) == len(coh[0, 0]))
    
    return freq, coh