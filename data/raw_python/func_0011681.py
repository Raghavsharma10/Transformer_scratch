def notch(ts, freq_hz, bandwidth_hz=1.0):
    """notch filter to remove remove a particular frequency
    Adapted from code by Sturla Molden
    """
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    channels = ts.shape[1]
    fs = (len(ts) - 1.0) / (ts.tspan[-1] - ts.tspan[0])
    nyq = 0.5 * fs
    freq = freq_hz/nyq
    bandwidth = bandwidth_hz/nyq
    R = 1.0 - 3.0*(bandwidth/2.0)
    K = ((1.0 - 2.0*R*np.cos(np.pi*freq) + R**2) /
         (2.0 - 2.0*np.cos(np.pi*freq)))
    b, a = np.zeros(3), np.zeros(3)
    a[0] = 1.0
    a[1] = -2.0*R*np.cos(np.pi*freq)
    a[2] = R**2
    b[0] = K
    b[1] = -2*K*np.cos(np.pi*freq)
    b[2] = K
    if not np.all(np.abs(np.roots(a)) < 1.0):
        raise ValueError('Filter will not be stable with these values.')
    dtype = ts.dtype
    output = np.zeros((len(ts), channels), dtype)
    for i in range(channels):
        output[:, i] = signal.filtfilt(b, a, ts[:, i])
    if orig_ndim is 1:
        output = output[:, 0]
    return Timeseries(output, ts.tspan, labels=ts.labels)