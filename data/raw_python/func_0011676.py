def plot(dts, title=None, points=None, show=True):
    """Plot a distributed timeseries
    Args: 
      dts (DistTimeseries)
      title (str, optional)
      points (int, optional): Limit the number of time points plotted. 
        If specified, will downsample to use this total number of time points, 
        and only fetch back the necessary points to the client for plotting.
    Returns: 
      fig
    """
    if points is not None and len(dts.tspan) > points:
        # then downsample  (TODO: use interpolation)
        ix = np.linspace(0, len(dts.tspan) - 1, points).astype(np.int64)
        dts = dts[ix, ...]
    ts = distob.gather(dts)
    return ts.plot(title, show)