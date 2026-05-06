def first_return_times(dts, c=None, d=0.0):
    """For an ensemble of time series, return the set of all time intervals
    between successive returns to value c for all instances in the ensemble.
    If c is not given, the default is the mean across all times and across all
    time series in the ensemble.

    Args:
      dts (DistTimeseries)

      c (float): Optional target value (default is the ensemble mean value)

      d (float): Optional min distance from c to be attained between returns

    Returns:
      array of time intervals (Can take the mean of these to estimate the
      expected first return time for the whole ensemble)
    """
    if c is None:
        c = dts.mean()
    vmrt = distob.vectorize(analyses1.first_return_times)
    all_intervals = vmrt(dts, c, d)
    if hasattr(type(all_intervals), '__array_interface__'):
        return np.ravel(all_intervals)
    else:
        return np.hstack([distob.gather(ilist) for ilist in all_intervals])