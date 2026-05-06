def hilbert_phase(ts):
    """Phase of the analytic signal, using the Hilbert transform"""
    output = np.angle(signal.hilbert(signal.detrend(ts, axis=0), axis=0))
    return Timeseries(output, ts.tspan, labels=ts.labels)