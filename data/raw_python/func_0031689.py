def crosscorrfunc(freq, cross):
    """
    Calculate crosscorrelation function(s) for given cross spectra.


    Parameters
    ----------
    freq : numpy.ndarray
        1 dimensional array of frequencies.
    cross : numpy.ndarray 
        2 dimensional array of cross spectra, 1st axis units, 2nd axis units,
        3rd axis frequencies.


    Returns
    -------
    time : tuple
        1 dim numpy.ndarray of times.
    crossf : tuple
        3 dim numpy.ndarray, crosscorrelation functions,
        1st axis first unit, 2nd axis second unit, 3rd axis times.

    """

    tbin = 1. / (2. * np.max(freq)) * 1e3  # tbin in ms
    time = np.arange(-len(freq) / 2. + 1, len(freq) / 2. + 1) * tbin
    # T = max(time)
    multidata = False
    # check whether cross contains many cross spectra
    if len(np.shape(cross)) > 1:
        multidata = True
    if multidata:
        N = len(cross)
        crossf = np.zeros((N, N, len(freq)))
        for i in range(N):
            for j in range(N):
                raw_crossf = np.real(np.fft.ifft(cross[i, j]))
                mid = int(len(raw_crossf) / 2.)
                crossf[i, j] = np.hstack(
                    [raw_crossf[mid + 1:], raw_crossf[:mid + 1]])
        assert(len(time) == len(crossf[0, 0]))
    else:
        raw_crossf = np.real(np.fft.ifft(cross))
        mid = int(len(raw_crossf) / 2.)
        crossf = np.hstack([raw_crossf[mid + 1:], raw_crossf[:mid + 1]])
        assert(len(time) == len(crossf))
    # crossf *= T*1e-3 # normalization happens in cross spectrum
    return time, crossf