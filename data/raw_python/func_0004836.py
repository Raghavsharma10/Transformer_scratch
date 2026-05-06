def line3d(
    x, y, z, label=None, color=None, width=None, dash=None, opacity=None, mode='lines+markers'
):
    """Create a 3d line chart."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    assert x.shape == y.shape
    assert y.shape == z.shape
    lineattr = {}
    if color:
        lineattr['color'] = color
    if width:
        lineattr['width'] = width
    if dash:
        lineattr['dash'] = dash
    if y.ndim == 2:
        if not hasattr(label, '__iter__'):
            if label is None:
                label = _labels()
            else:
                label = _labels(label)
        data = [
            go.Scatter3d(x=xx, y=yy, z=zz, name=ll, line=lineattr, mode=mode, opacity=opacity)
            for ll, xx, yy, zz in zip(label, x.T, y.T, z.T)
        ]
    else:
        data = [go.Scatter3d(x=x, y=y, z=z, name=label, line=lineattr, mode=mode, opacity=opacity)]
    return Chart(data=data)