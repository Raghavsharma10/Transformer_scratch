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
    variability, allchannels_epochs = ts.epochs_distributed(
            variability, threshold, minlength, plot=False)
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
        allchannels_epochs = [allchannels_epochs]
        variability = variability[:, np.newaxis]
    channels = ts.shape[1]
    dt = (1.0*ts.tspan[-1] - ts.tspan[0]) / (len(ts) - 1)
    starts = [(e[0], 1) for channel in allchannels_epochs for e in channel]
    ends = [(e[1], -1) for channel in allchannels_epochs for e in channel]
    all = sorted(starts + ends)
    joint_epochs = []
    in_joint_epoch = False
    joint_start = 0.0
    inside_count = 0
    for bound in all:
        inside_count += bound[1]
        if not in_joint_epoch and 1.0*inside_count/channels >= proportion:
            in_joint_epoch = True
            joint_start = bound[0]
        if in_joint_epoch and 1.0*inside_count/channels < proportion:
            in_joint_epoch = False
            joint_end = bound[0]
            if (joint_end - joint_start)*dt >= minlength:
                joint_epochs.append((joint_start, joint_end))
    if plot:
        joint_epochs_repeated = [joint_epochs] * channels
        _plot_variability(ts, variability, threshold, joint_epochs_repeated)
    return (variability, joint_epochs)