def crossing_times(ts, c=0.0, d=0.0):
    """For a single variable timeseries, find the times at which the
    value crosses ``c`` from above or below. Can optionally set a non-zero
    ``d`` to impose the condition that the value must wander at least ``d`` 
    units away from ``c`` between crossings.

    If the timeseries begins (or ends) exactly at ``c``, then time zero 
    (or the ending time) is also included as a crossing event, 
    so that the boundaries of the first and last excursions are included.

    If the actual crossing time falls between two time steps, linear
    interpolation is used to estimate the crossing time.

    Args:
      ts: Timeseries (single variable)

      c (float): Critical value at which to report crossings.

      d (float): Optional min distance from c to be attained between crossings.

    Returns:
      array of float
    """
    #TODO support multivariate time series
    ts = ts.squeeze()
    if ts.ndim is not 1:
        raise ValueError('Currently can only use on single variable timeseries')

    # Translate to put the critical value at zero:
    ts = ts - c

    tsa = ts[0:-1]
    tsb = ts[1:]
    # Time indices where phase crosses or reaches zero from below or above
    zc = np.nonzero((tsa < 0) & (tsb >= 0) | (tsa > 0) & (tsb <= 0))[0] + 1
    # Estimate crossing time interpolated linearly within a single time step
    va = ts[zc-1]
    vb = ts[zc]
    ct = (np.abs(vb)*ts.tspan[zc-1] +
          np.abs(va)*ts.tspan[zc]) / np.abs(vb - va) # denominator always !=0
    # Also include starting time if we started exactly at zero
    if ts[0] == 0.0:
        zc = np.r_[np.array([0]), zc]
        ct = np.r_[np.array([ts.tspan[0]]), ct]

    if d == 0.0 or ct.shape[0] is 0:
        return ct

    # Time indices where value crosses c+d or c-d:
    dc = np.nonzero((tsa < d) & (tsb >= d) | (tsa > -d) & (tsb <= -d))[0] + 1
    # Select those zero-crossings separated by at least one d-crossing
    splice = np.searchsorted(dc, zc)
    which_zc = np.r_[np.array([0]), np.nonzero(splice[0:-1] - splice[1:])[0] +1]
    return ct[which_zc]