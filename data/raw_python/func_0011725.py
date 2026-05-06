def epochs_distributed(ts, variability=None, threshold=0.0, minlength=1.0, 
                       plot=True):
    """Same as `epochs()`, but computes channels in parallel for speed.

    (Note: This requires an IPython cluster to be started first, 
           e.g. on a workstation type 'ipcluster start')

    Identify "stationary" epochs within a time series, based on a 
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
    import distob
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    if variability is None:
        dts = distob.scatter(ts, axis=1)
        vepochs = distob.vectorize(epochs)
        results = vepochs(dts, None, threshold, minlength, plot=False)
    else: 
        def f(pair):
            return epochs(pair[0], pair[1], threshold, minlength, plot=False)
        allpairs = [(ts[:, i], variability[:, i]) for i in range(ts.shape[1])]
        vf = distob.vectorize(f)
        results = vf(allpairs)
    vars, allchannels_epochs = zip(*results)
    variability = distob.hstack(vars)
    if plot:
        _plot_variability(ts, variability, threshold, allchannels_epochs)
    return (variability, allchannels_epochs)