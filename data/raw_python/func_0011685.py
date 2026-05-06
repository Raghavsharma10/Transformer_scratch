def cwt(ts, freqs=np.logspace(0, 2), wavelet=cwtmorlet, plot=True):
    """Continuous wavelet transform
    Note the full results can use a huge amount of memory at 64-bit precision

    Args:
      ts: Timeseries of m variables, shape (n, m). Assumed constant timestep.
      freqs: list of frequencies (in Hz) to use for the tranform. 
        (default is 50 frequency bins logarithmic from 1Hz to 100Hz)
      wavelet: the wavelet to use. may be complex. see scipy.signal.wavelets
      plot: whether to plot time-resolved power spectrum

    Returns: 
      coefs: Continuous wavelet transform output array, shape (n,len(freqs),m)
    """
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    channels = ts.shape[1]
    fs = (len(ts) - 1.0) / (1.0*ts.tspan[-1] - ts.tspan[0])
    x = signal.detrend(ts, axis=0)
    dtype = wavelet(fs/freqs[0], fs/freqs[0]).dtype
    coefs = np.zeros((len(ts), len(freqs), channels), dtype)
    for i in range(channels):
        coefs[:, :, i] = roughcwt(x[:, i], cwtmorlet, fs/freqs).T
    if plot:
        _plot_cwt(ts, coefs, freqs)
    if orig_ndim is 1:
        coefs = coefs[:, :, 0]
    return coefs