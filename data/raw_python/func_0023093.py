def curve3_bezier(p1, p2, p3):
    """
    Generate the vertices for a quadratic Bezier curve.

    The vertices returned by this function can be passed to a LineVisual or
    ArrowVisual.

    Parameters
    ----------
    p1 : array
        2D coordinates of the start point
    p2 : array
        2D coordinates of the first curve point
    p3 : array
        2D coordinates of the end point

    Returns
    -------
    coords : list
        Vertices for the Bezier curve.

    See Also
    --------
    curve4_bezier

    Notes
    -----
    For more information about Bezier curves please refer to the `Wikipedia`_
    page.

    .. _Wikipedia: https://en.wikipedia.org/wiki/B%C3%A9zier_curve
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    points = []
    _curve3_recursive_bezier(points, x1, y1, x2, y2, x3, y3)

    dx, dy = points[0][0] - x1, points[0][1] - y1
    if (dx * dx + dy * dy) > 1e-10:
        points.insert(0, (x1, y1))

    dx, dy = points[-1][0] - x3, points[-1][1] - y3
    if (dx * dx + dy * dy) > 1e-10:
        points.append((x3, y3))

    return np.array(points).reshape(len(points), 2)