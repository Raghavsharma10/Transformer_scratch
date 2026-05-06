def fill_between(
    x=None,
    ylow=None,
    yhigh=None,
    label=None,
    color=None,
    width=None,
    dash=None,
    opacity=None,
    mode='lines+markers',
    **kargs
):
    """Fill between `ylow` and `yhigh`.

    Parameters
    ----------
    x : array-like, optional
    ylow : TODO, optional
    yhigh : TODO, optional

    Returns
    -------
    Chart

    """
    plot = line(
        x=x,
        y=ylow,
        label=label,
        color=color,
        width=width,
        dash=dash,
        opacity=opacity,
        mode=mode,
        fill=None,
        **kargs
    )
    plot += line(
        x=x,
        y=yhigh,
        label=label,
        color=color,
        width=width,
        dash=dash,
        opacity=opacity,
        mode=mode,
        fill='tonexty',
        **kargs
    )
    return plot