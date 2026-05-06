def variability_fp(ts, freqs=None, ncycles=6, plot=True):
    """Example variability function.
    Gives two continuous, time-resolved measures of the variability of a
    time series, ranging between -1 and 1. 
    The two measures are based on variance of the centroid frequency and 
    variance of the height of the spectral peak, respectively.
    (Centroid frequency meaning the power-weighted average frequency)
    These measures are calculated over sliding time windows of variable size.
    See also: Blenkinsop et al. (2012) The dynamic evolution of focal-onset 
              epilepsies - combining theoretical and clinical observations
    Args:
      ts  Timeseries of m variables, shape (n, m). Assumed constant timestep.
      freqs   (optional) List of frequencies to examine. If None, defaults to
              50 frequency bands ranging 1Hz to 60Hz, logarithmically spaced.
      ncycles  Window size, in number of cycles of the centroid frequency.
      plot  bool  Whether to display the output

    Returns:
      variability   Timeseries of shape (n, m, 2)  
                    variability[:, :, 0] gives a measure of variability 
                    between -1 and 1 based on variance of centroid frequency.
                    variability[:, :, 1] gives a measure of variability 
                    between -1 and 1 based on variance of maximum power.
    """
    if freqs is None:
        freqs = np.logspace(np.log10(1.0), np.log10(60.0), 50)
    else:
        freqs = np.array(freqs)
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    channels = ts.shape[1]
    n = len(ts)
    dt = (1.0*ts.tspan[-1] - ts.tspan[0]) / (n - 1)
    fs = 1.0 / dt
    dtype = ts.dtype
    # Estimate time-resolved power spectra using continuous wavelet transform
    coefs = ts.cwt(freqs, wavelet=cwtmorlet, plot=False)
    # this is a huge array so try to do operations in place
    powers = np.square(np.abs(coefs, coefs), coefs).real.astype(dtype, 
                                                                copy=False)
    del coefs
    max_power = np.max(powers, axis=1)
    total_power = np.sum(powers, axis=1, keepdims=True)
    rel_power = np.divide(powers, total_power, powers)
    del powers
    centroid_freq = np.tensordot(freqs, rel_power, axes=(0, 1))  # shape (n, m)
    del rel_power
    # hw is half window size (in number of samples)
    hw = np.int64(np.ceil(0.5 * ncycles * fs / centroid_freq))  # shape (n, m)
    allchannels_variability = np.zeros((n, channels, 2), dtype) # output array
    for i in range(channels):
        logvar_centfreq = np.zeros(n, dtype)
        logvar_maxpower = np.zeros(n, dtype)
        for j in range(n):
            # compute variance of two chosen signal properties over a 
            # window of 2*hw+1 samples centered on sample number j
            wstart = j - hw[j, i]
            wend = j + hw[j, i]
            if wstart >= 0 and wend < n:
                logvar_centfreq[j] = np.log(centroid_freq[wstart:wend+1].var())
                logvar_maxpower[j] = np.log(max_power[wstart:wend+1].var())
            else:
                logvar_centfreq[j] = np.nan
                logvar_maxpower[j] = np.nan
        allchannels_variability[:, i, 0] = _rescale(logvar_centfreq)
        allchannels_variability[:, i, 1] = _rescale(logvar_maxpower)
    allchannels_variability = Timeseries(allchannels_variability, 
                                         ts.tspan, labels=ts.labels)
    if plot:
        _plot_variability(ts, allchannels_variability)
    return allchannels_variability