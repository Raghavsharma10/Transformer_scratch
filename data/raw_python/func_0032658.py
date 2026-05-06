def isPointInsidePolygon(x, y, vertices_x, vertices_y):
    """Check if a given point is inside a polygon.

    Parameters vertices_x[] and vertices_y[] define the polygon.
    The number of array elements is equal to number of vertices of the polygon.
    This function works for convex and concave polygons.

    Parameters
    ----------
    vertices_x, vertices_y : lists or arrays of floats
        Vertices that define the polygon.

    x, y : float
        Coordinates of the point to check.

    Returns
    -------
    inside : bool
        `True` if the point is inside the polygon.
    """
    inside = False
    for i in range(len(vertices_x)):
        j = i - 1
        if ((vertices_x[i] > x) != (vertices_x[j] > x)):
            if (y < (x - vertices_x[i]) *
                    (vertices_y[i] - vertices_y[j]) /
                    (vertices_x[i] - vertices_x[j]) +
                    vertices_y[i]):
                inside = not inside
    return inside