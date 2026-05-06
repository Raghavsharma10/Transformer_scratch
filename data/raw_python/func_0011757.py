def autocorrelation(ts, normalized=False, unbiased=False):
    """
    Returns the discrete, linear convolution of a time series with itself, 
    optionally using unbiased normalization. 

    N.B. Autocorrelation estimates are necessarily inaccurate for longer lags,
    as there are less pairs of points to convolve separated by that lag.
    Therefore best to throw out the results except for shorter lags, e.g. 
    keep lags from tau=0 up to one quarter of the total time series length.

    Args:
      normalized (boolean): If True, the time series will first be normalized
        to a mean of 0 and variance of 1. This gives autocorrelation 1 at
        zero lag.

      unbiased (boolean): If True, the result at each lag m will be scaled by
        1/(N-m). This gives an unbiased estimation of the autocorrelation of a
        stationary process from a finite length sample.

    Ref: S. J. Orfanidis (1996) "Optimum Signal Processing", 2nd Ed.
    """
    ts = np.squeeze(ts)
    if ts.ndim <= 1:
        if normalized:
            ts = (ts - ts.mean())/ts.std()
        N = ts.shape[0]
        ar = np.asarray(ts)
        acf = np.correlate(ar, ar, mode='full')
        outlen = (acf.shape[0] + 1) / 2
        acf = acf[(outlen - 1):]
        if unbiased:
            factor = np.array([1.0/(N - m) for m in range(0, outlen)])
            acf = acf * factor
        dt = (ts.tspan[-1] - ts.tspan[0]) / (len(ts) - 1.0)
        lags = np.arange(outlen)*dt
        return Timeseries(acf, tspan=lags, labels=ts.labels)
    else:
        # recursively handle arrays of dimension > 1
        lastaxis = ts.ndim - 1
        m = ts.shape[lastaxis]
        acfs = [ts[...,i].autocorrelation(normalized, unbiased)[...,np.newaxis]
                for i in range(m)]
        res = distob.concatenate(acfs, axis=lastaxis)
        res.labels[lastaxis] = ts.labels[lastaxis]
        return res