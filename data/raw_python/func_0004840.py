def fill_zero(
    x=None,
    y=None,
    label=None,
    color=None,
    width=None,
    dash=None,
    opacity=None,
    mode='lines+markers',
    **kargs
):
    """Fill to zero.

    Parameters
    ----------
    x : array-like, optional
    y : TODO, optional
    label : TODO, optional

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
        fill='tozeroy',
        **kargs
    )