def hist(x, mode='overlay', label=None, opacity=None, horz=False, histnorm=None):
    """Histogram.

    Parameters
    ----------
    x : array-like
    mode : str, optional
    label : TODO, optional
    opacity : float, optional
    horz : bool, optional
    histnorm : None, "percent", "probability", "density", "probability density", optional
        Specifies the type of normalization used for this histogram trace.
        If ``None``, the span of each bar corresponds to the number of occurrences
        (i.e. the number of data points lying inside the bins). If "percent",
        the span of each bar corresponds to the percentage of occurrences with
        respect to the total number of sample points (here, the sum of all bin
        area equals 100%). If "density", the span of each bar corresponds to the
        number of occurrences in a bin divided by the size of the bin interval
        (here, the sum of all bin area equals the total number of sample
        points). If "probability density", the span of each bar corresponds to
        the probability that an event will fall into the corresponding bin
        (here, the sum of all bin area equals 1).

    Returns
    -------
    Chart

    """
    x = np.atleast_1d(x)
    if horz:
        kargs = dict(y=x)
    else:
        kargs = dict(x=x)
    layout = dict(barmode=mode)
    data = [go.Histogram(opacity=opacity, name=label, histnorm=histnorm, **kargs)]
    return Chart(data=data, layout=layout)