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
    if ts.ndim <= 2:
        return analyses1.epochs_distributed(
                ts, variability, threshold, minlength, plot)
    else:
        return distob.vectorize(analyses1.epochs)(
                ts, variability, threshold, minlength, plot)