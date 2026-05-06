def cwt_distributed(ts, freqs=np.logspace(0, 2), wavelet=cwtmorlet, plot=True):
    """Continuous wavelet transform using distributed computation.
    (Currently just splits the data by channel. TODO split it further.)
    Note: this function requires an IPython cluster to be started first.

    Args:
      ts: Timeseries of m variables, shape (n, m). Assumed constant timestep.
      freqs: list of frequencies (in Hz) to use for the tranform. 
        (default is 50 frequency bins logarithmic from 1Hz to 100Hz)
      wavelet: the wavelet to use. may be complex. see scipy.signal.wavelets
      plot: whether to plot time-resolved power spectrum

    Returns: 
      coefs: Continuous wavelet transform output array, shape (n,len(freqs),m)
    """
    if ts.ndim is 1 or ts.shape[1] is 1:
        return cwt(ts, freqs, wavelet, plot)
    import distob
    vcwt = distob.vectorize(cwt)
    coefs = vcwt(ts, freqs, wavelet, plot=False)
    if plot:
        _plot_cwt(ts, coefs, freqs)
    return coefs