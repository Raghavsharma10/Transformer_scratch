def curve4_bezier(p1, p2, p3, p4):
    """
    Generate the vertices for a third order Bezier curve.

    The vertices returned by this function can be passed to a LineVisual or
    ArrowVisual.

    Parameters
    ----------
    p1 : array
        2D coordinates of the start point
    p2 : array
        2D coordinates of the first curve point
    p3 : array
        2D coordinates of the second curve point
    p4 : array
        2D coordinates of the end point

    Returns
    -------
    coords : list
        Vertices for the Bezier curve.

    See Also
    --------
    curve3_bezier

    Notes
    -----
    For more information about Bezier curves please refer to the `Wikipedia`_
    page.

    .. _Wikipedia: https://en.wikipedia.org/wiki/B%C3%A9zier_curve
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    points = []
    _curve4_recursive_bezier(points, x1, y1, x2, y2, x3, y3, x4, y4)

    dx, dy = points[0][0] - x1, points[0][1] - y1
    if (dx * dx + dy * dy) > 1e-10:
        points.insert(0, (x1, y1))
    dx, dy = points[-1][0] - x4, points[-1][1] - y4
    if (dx * dx + dy * dy) > 1e-10:
        points.append((x4, y4))

    return np.array(points).reshape(len(points), 2)