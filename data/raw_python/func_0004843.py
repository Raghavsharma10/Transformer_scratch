def surface(x, y, z):
    """Surface plot.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    z : array-like, optional

    Returns
    -------
    Chart

    """
    data = [go.Surface(x=x, y=y, z=z)]
    return Chart(data=data)