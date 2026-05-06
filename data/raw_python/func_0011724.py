def epochs(ts, variability=None, threshold=0.0, minlength=1.0, plot=True):
    """Identify "stationary" epochs within a time series, based on a 
    continuous measure of variability.
    Epochs are defined to contain the points of minimal variability, and to 
    extend as wide as possible with variability not exceeding the threshold.

    Args:
      ts  Timeseries of m variables, shape (n, m). 
      variability  (optional) Timeseries of shape (n, m, q),  giving q scalar 
                   measures of the variability of timeseries `ts` near each 
                   point in time. (if None, we will use variability_fp())
                   Epochs require the mean of these to be below the threshold.
      threshold   The maximum variability permitted in stationary epochs.
      minlength   Shortest acceptable epoch length (in seconds)
      plot  bool  Whether to display the output

    Returns: (variability, allchannels_epochs) 
      variability: as above
      allchannels_epochs: (list of) list of tuples
      For each variable, a list of tuples (start, end) that give the 
      starting and ending indices of stationary epochs.
      (epochs are inclusive of start point but not the end point)
    """
    if variability is None:
        variability = ts.variability_fp(plot=False)
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    if variability.ndim is 1:
        variability = variability[:, np.newaxis, np.newaxis]
    elif variability.ndim is 2:
        variability = variability[:, np.newaxis, :]
    channels = ts.shape[1]
    n = len(ts)
    dt = (1.0*ts.tspan[-1] - ts.tspan[0]) / (n - 1)
    fs = 1.0 / dt
    allchannels_epochs = []
    for i in range(channels):
        v = variability[:, i, :]
        v = np.nanmean(v, axis=1) # mean of q different variability measures
        # then smooth the variability with a low-pass filter
        nonnan_ix = np.nonzero(~np.isnan(v))[0]
        nonnans = slice(nonnan_ix.min(), nonnan_ix.max())
        crit_freq = 1.0 # Hz
        b, a = signal.butter(3, 2.0 * crit_freq / fs)
        #v[nonnans] = signal.filtfilt(b, a, v[nonnans])
        v[nonnan_ix] = signal.filtfilt(b, a, v[nonnan_ix])
        # find all local minima of the variability not exceeding the threshold
        m = v[1:-1]
        l = v[0:-2]
        r = v[2:]
        minima = np.nonzero(~np.isnan(m) & ~np.isnan(l) & ~np.isnan(r) &
                            (m <= threshold) & (m-l < 0) & (r-m > 0))[0] + 1
        if len(minima) is 0:
            print(u'Channel %d: no epochs found using threshold %g' % (
                i, threshold))
            allchannels_epochs.append([])
        else:
            # Sort the list of minima by ascending variability
            minima = minima[np.argsort(v[minima])]
            epochs = []
            for m in minima:
                # Check this minimum is not inside an existing epoch
                overlap = False
                for e in epochs:
                    if m >= e[0] and m <= e[1]:
                        overlap = True
                        break
                if not overlap:
                    # Get largest subthreshold interval surrounding the minimum
                    startix = m - 1
                    endix = m + 1
                    for startix in range(m - 1, 0, -1):
                        if np.isnan(v[startix]) or v[startix] > threshold:
                            startix += 1
                            break
                    for endix in range(m + 1, len(v), 1):
                        if np.isnan(v[endix]) or v[endix] > threshold:
                            break
                    if (endix - startix) * dt >= minlength: 
                        epochs.append((startix, endix))
            allchannels_epochs.append(epochs)
    if plot:
        _plot_variability(ts, variability, threshold, allchannels_epochs)
    if orig_ndim is 1:
        allchannels_epochs = allchannels_epochs[0]
    return (variability, allchannels_epochs)