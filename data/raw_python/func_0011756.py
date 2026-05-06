def first_return_times(ts, c=None, d=0.0):
    """For a single variable time series, first wait until the time series
    attains the value c for the first time. Then record the time intervals 
    between successive returns to c. If c is not given, the default is the mean
    of the time series.
    
    Args:
      ts: Timeseries (single variable)

      c (float): Optional target value (default is the mean of the time series)

      d (float): Optional min distance from c to be attained between returns

    Returns:
      array of time intervals (Can take the mean of these to estimate the
      expected first return time)
    """
    ts = np.squeeze(ts)
    if c is None:
        c = ts.mean()
    if ts.ndim <= 1:
        return np.diff(ts.crossing_times(c, d))
    else:
        return np.hstack(
            [ts[..., i].first_return_times(c, d) for i in range(ts.shape[-1])])