def autocorrfunc(freq, power):
    """
    Calculate autocorrelation function(s) for given power spectrum/spectra.


    Parameters
    ----------
    freq : numpy.ndarray
        1 dimensional array of frequencies.
    power : numpy.ndarray
        2 dimensional power spectra, 1st axis units, 2nd axis frequencies.


    Returns
    -------
    time : tuple
        1 dim numpy.ndarray of times.
    autof : tuple
        2 dim numpy.ndarray; autocorrelation functions, 1st axis units,
        2nd axis times.

    """
    tbin = 1. / (2. * np.max(freq)) * 1e3  # tbin in ms
    time = np.arange(-len(freq) / 2. + 1, len(freq) / 2. + 1) * tbin
    # T = max(time)
    multidata = False
    if len(np.shape(power)) > 1:
        multidata = True
    if multidata:
        N = len(power)
        autof = np.zeros((N, len(freq)))
        for i in range(N):
            raw_autof = np.real(np.fft.ifft(power[i]))
            mid = int(len(raw_autof) / 2.)
            autof[i] = np.hstack([raw_autof[mid + 1:], raw_autof[:mid + 1]])
        assert(len(time) == len(autof[0]))
    else:
        raw_autof = np.real(np.fft.ifft(power))
        mid = int(len(raw_autof) / 2.)
        autof = np.hstack([raw_autof[mid + 1:], raw_autof[:mid + 1]])
        assert(len(time) == len(autof))
    # autof *= T*1e-3 # normalization is done in powerspec()
    return time, autof