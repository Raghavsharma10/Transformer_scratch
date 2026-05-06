def horizontal(y, xmin=0, xmax=1, color=None, width=None, dash=None, opacity=None):
    """Draws a horizontal line from `xmin` to `xmax`.

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
        shapes=[dict(type='line', x0=xmin, x1=xmax, y0=y, y1=y, opacity=opacity, line=lineattr)]
    )
    return Chart(layout=layout)