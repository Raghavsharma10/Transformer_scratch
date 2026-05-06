def scatter(
    x=None,
    y=None,
    label=None,
    color=None,
    width=None,
    dash=None,
    opacity=None,
    markersize=6,
    yaxis=1,
    fill=None,
    text="",
    mode='markers',
):
    """Draws dots.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    label : array-like, optional

    Returns
    -------
    Chart

    """
    return line(
        x=x,
        y=y,
        label=label,
        color=color,
        width=width,
        dash=dash,
        opacity=opacity,
        mode=mode,
        yaxis=yaxis,
        fill=fill,
        text=text,
        markersize=markersize,
    )