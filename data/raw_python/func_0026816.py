def _mid(pt1, pt2):
    """
    (Point, Point) -> Point
    Return the point that lies in between the two input points.
    """
    (x0, y0), (x1, y1) = pt1, pt2
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)