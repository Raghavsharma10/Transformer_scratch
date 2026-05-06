def bar(x=None, y=None, label=None, mode='group', yaxis=1, opacity=None):
    """Create a bar chart.

    Parameters
    ----------
    x : array-like, optional
    y : TODO, optional
    label : TODO, optional
    mode : 'group' or 'stack', default 'group'
    opacity : TODO, optional

    Returns
    -------
    Chart
        A Chart with bar graph data.

    """
    assert x is not None or y is not None, "x or y must be something"
    yn = 'y' + str(yaxis)
    if y is None:
        y = x
        x = None
    if x is None:
        x = np.arange(len(y))
    else:
        x = _try_pydatetime(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if y.ndim == 2:
        if not hasattr(label, '__iter__'):
            if label is None:
                label = _labels()
            else:
                label = _labels(label)
        data = [go.Bar(x=x, y=yy, name=ll, yaxis=yn, opacity=opacity) for ll, yy in zip(label, y.T)]
    else:
        data = [go.Bar(x=x, y=y, name=label, yaxis=yn, opacity=opacity)]
    if yaxis == 1:
        return Chart(data=data, layout={'barmode': mode})

    return Chart(data=data, layout={'barmode': mode, 'yaxis' + str(yaxis): dict(overlaying='y')})