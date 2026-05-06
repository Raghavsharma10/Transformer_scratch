def _remove_pi_crossings(ts):
    """For each variable in the Timeseries, checks whether it represents
    a phase variable ranging from -pi to pi. If so, set all points where the
    phase crosses pi to 'nan' so that spurious lines will not be plotted.

    If ts does not need adjustment, then return ts. 
    Otherwise return a modified copy.
    """
    orig_ts = ts
    if ts.ndim is 1:
        ts = ts[:, np.newaxis, np.newaxis]
    elif ts.ndim is 2:
        ts = ts[:, np.newaxis]
    # Get the indices of those variables that have range of approx -pi to pi
    tsmax = ts.max(axis=0)
    tsmin = ts.min(axis=0)
    phase_vars = np.transpose(np.nonzero((np.abs(tsmax - np.pi) < 0.01) & 
                                         (np.abs(tsmin + np.pi) < 0.01)))
    if len(phase_vars) is 0:
        return orig_ts
    else:
        ts = ts.copy()
        for v in phase_vars:
            ts1 = np.asarray(ts[:, v[0], v[1]]) # time series of single variable
            ts1a = ts1[0:-1]
            ts1b = ts1[1:]
            p2 = np.pi/2
            # Find time indices where phase crosses pi. Set those values to nan.
            pc = np.nonzero((ts1a > p2) & (ts1b < -p2) | 
                            (ts1a < -p2) & (ts1b > p2))[0] + 1
            ts1[pc] = np.nan
            ts[:, v[0], v[1]] = ts1
        return ts