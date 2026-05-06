def epochs_joint(ts, variability=None, threshold=0.0, minlength=1.0,
                 proportion=0.75, plot=True):
    """Identify epochs within a multivariate time series where at least a 
    certain proportion of channels are "stationary", based on a previously 
    computed variability measure.

    (Note: This requires an IPython cluster to be started first, 
     e.g. on a workstation type 'ipcluster start')

    Args:
      ts  Timeseries of m variables, shape (n, m). 
      variability  (optional) Timeseries of shape (n, m),  giving a scalar 
                   measure of the variability of timeseries `ts` near each 
                   point in time. (if None, we will use variability_fp())
      threshold   The maximum variability permitted in stationary epochs.
      minlength   Shortest acceptable epoch length (in seconds)
      proportion  Require at least this fraction of channels to be "stationary"
      plot  bool  Whether to display the output

    Returns: (variability, joint_epochs)
      joint_epochs: list of tuples
      A list of tuples (start, end) that give the starting and ending indices 
      of time epochs that are stationary for at least `proportion` of channels.
      (epochs are inclusive of start point but not the end point)
    """
    if ts.ndim <= 2:
        return analyses1.epochs_joint(
                ts, variability, threshold, minlength, plot)
    else:
        return distob.vectorize(analyses1.epochs_joint)(
                ts, variability, threshold, minlength, plot)