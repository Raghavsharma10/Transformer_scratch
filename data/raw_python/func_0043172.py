def Lomb_Scargle(data, precision, min_period, max_period, period_jobs=1):
    """
    Returns the period of *data* according to the
    `Lomb-Scargle periodogram <https://en.wikipedia.org/wiki/Least-squares_spectral_analysis#The_Lomb.E2.80.93Scargle_periodogram>`_.

    **Parameters**

    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Array containing columns *time*, *mag*, and (optional) *error*.
    precision : number
        Distance between contiguous frequencies in search-space.
    min_period : number
        Minimum period in search-space.
    max_period : number
        Maximum period in search-space.
    period_jobs : int, optional
        Number of simultaneous processes to use while searching. Only one
        process will ever be used, but argument is included to conform to
        *periodogram* standards of :func:`find_period` (default 1).

    **Returns**

    period : number
        The period of *data*.
    """
    time, mags, *err = data.T
    scaled_mags = (mags-mags.mean())/mags.std()
    minf, maxf = 2*np.pi/max_period, 2*np.pi/min_period
    freqs = np.arange(minf, maxf, precision)
    pgram = lombscargle(time, scaled_mags, freqs)

    return 2*np.pi/freqs[np.argmax(pgram)]