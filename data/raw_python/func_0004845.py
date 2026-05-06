def hist2d(x, y, label=None, opacity=None):
    """2D Histogram.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    label : TODO, optional
    opacity : float, optional

    Returns
    -------
    Chart

    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    data = [go.Histogram2d(x=x, y=y, opacity=opacity, name=label)]
    return Chart(data=data)