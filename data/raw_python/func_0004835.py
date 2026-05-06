def line(
    x=None,
    y=None,
    label=None,
    color=None,
    width=None,
    dash=None,
    opacity=None,
    mode='lines+markers',
    yaxis=1,
    fill=None,
    text="",
    markersize=6,
):
    """Draws connected dots.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    label : array-like, optional

    Returns
    -------
    Chart

    """
    assert x is not None or y is not None, "x or y must be something"
    yn = 'y' + str(yaxis)
    lineattr = {}
    if color:
        lineattr['color'] = color
    if width:
        lineattr['width'] = width
    if dash:
        lineattr['dash'] = dash
    if y is None:
        y = x
        x = None
    if x is None:
        x = np.arange(len(y))
    else:
        x = _try_pydatetime(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    assert x.shape[0] == y.shape[0]
    if y.ndim == 2:
        if not hasattr(label, '__iter__'):
            if label is None:
                label = _labels()
            else:
                label = _labels(label)
        data = [
            go.Scatter(
                x=x,
                y=yy,
                name=ll,
                line=lineattr,
                mode=mode,
                text=text,
                fill=fill,
                opacity=opacity,
                yaxis=yn,
                marker=dict(size=markersize, opacity=opacity),
            )
            for ll, yy in zip(label, y.T)
        ]
    else:
        data = [
            go.Scatter(
                x=x,
                y=y,
                name=label,
                line=lineattr,
                mode=mode,
                text=text,
                fill=fill,
                opacity=opacity,
                yaxis=yn,
                marker=dict(size=markersize, opacity=opacity),
            )
        ]
    if yaxis == 1:
        return Chart(data=data)

    return Chart(data=data, layout={'yaxis' + str(yaxis): dict(overlaying='y')})