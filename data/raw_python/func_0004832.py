def spark_shape(points, shapes, fill=None, color='blue', width=5, yindex=0, heights=None):
    """TODO: Docstring for spark.

    Parameters
    ----------
    points : array-like
    shapes : array-like
    fill : array-like, optional

    Returns
    -------
    Chart

    """
    assert len(points) == len(shapes) + 1
    data = [{'marker': {'color': 'white'}, 'x': [points[0], points[-1]], 'y': [yindex, yindex]}]

    if fill is None:
        fill = [False] * len(shapes)

    if heights is None:
        heights = [0.4] * len(shapes)

    lays = []
    for i, (shape, height) in enumerate(zip(shapes, heights)):
        if shape is None:
            continue
        if fill[i]:
            fillcolor = color
        else:
            fillcolor = 'white'
        lays.append(
            dict(
                type=shape,
                x0=points[i],
                x1=points[i + 1],
                y0=yindex - height,
                y1=yindex + height,
                xref='x',
                yref='y',
                fillcolor=fillcolor,
                line=dict(color=color, width=width),
            )
        )

    layout = dict(shapes=lays)

    return Chart(data=data, layout=layout)