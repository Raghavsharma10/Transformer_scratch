def heatmap(z, x=None, y=None, colorscale='Viridis'):
    """Create a heatmap.

    Parameters
    ----------
    z : TODO
    x : TODO, optional
    y : TODO, optional
    colorscale : TODO, optional

    Returns
    -------
    Chart


    """
    z = np.atleast_1d(z)
    data = [go.Heatmap(z=z, x=x, y=y, colorscale=colorscale)]
    return Chart(data=data)