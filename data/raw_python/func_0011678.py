def psd(ts, nperseg=1500, noverlap=1200, plot=True):
    """plot Welch estimate of power spectral density, using nperseg samples per
    segment, with noverlap samples overlap and Hamming window."""
    ts = ts.squeeze()
    if ts.ndim is 1:
        ts = ts.reshape((-1, 1))
    fs = (len(ts) - 1.0) / (ts.tspan[-1] - ts.tspan[0])
    window = signal.hamming(nperseg, sym=False)
    nfft = max(256, 2**np.int(np.log2(nperseg) + 1))
    freqs, pxx = signal.welch(ts, fs, window, nperseg, noverlap, nfft,
                              detrend='linear', axis=0)
    # Discard estimates for freq bins that are too low for the window size.
    # (require two full cycles to fit within the window)
    index = np.nonzero(freqs >= 2.0*fs/nperseg)[0][0]
    if index > 0:
        freqs = freqs[index:]
        pxx = pxx[index:]
    # Discard estimate for last freq bin as too high for Nyquist frequency:
    freqs = freqs[:-1]
    pxx = pxx[:-1]
    if plot is True:
        _plot_psd(ts, freqs, pxx)
    return freqs, pxx