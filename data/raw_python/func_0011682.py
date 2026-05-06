def hilbert(ts):
    """Analytic signal, using the Hilbert transform"""
    output = signal.hilbert(signal.detrend(ts, axis=0), axis=0)
    return Timeseries(output, ts.tspan, labels=ts.labels)