def vertical(x, ymin=0, ymax=1, color=None, width=None, dash=None, opacity=None):
    """Draws a vertical line from `ymin` to `ymax`.

    Parameters
    ----------
    xmin : int, optional
    xmax : int, optional
    color : str, optional
    width : number, optional

    Returns
    -------
    Chart

    """
    lineattr = {}
    if color:
        lineattr['color'] = color
    if width:
        lineattr['width'] = width
    if dash:
        lineattr['dash'] = dash

    layout = dict(
        shapes=[dict(type='line', x0=x, x1=x, y0=ymin, y1=ymax, opacity=opacity, line=lineattr)]
    )
    return Chart(layout=layout)