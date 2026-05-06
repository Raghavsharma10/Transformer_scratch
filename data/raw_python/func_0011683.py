def hilbert_amplitude(ts):
    """Amplitude of the analytic signal, using the Hilbert transform"""
    output = np.abs(signal.hilbert(signal.detrend(ts, axis=0), axis=0))
    return Timeseries(output, ts.tspan, labels=ts.labels)