def _tValueForPointOnCubicCurve(point, cubicCurve, isHorizontal=0):
    """
    Finds a t value on a curve from a point.
    The points must be originaly be a point on the curve.
    This will only back trace the t value, needed to split the curve in parts
    """
    pt1, pt2, pt3, pt4 = cubicCurve
    a, b, c, d = bezierTools.calcCubicParameters(pt1, pt2, pt3, pt4)
    solutions = bezierTools.solveCubic(a[isHorizontal], b[isHorizontal], c[isHorizontal],
        d[isHorizontal] - point[isHorizontal])
    solutions = [t for t in solutions if 0 <= t < 1]
    if not solutions and not isHorizontal:
        # can happen that a horizontal line doens intersect, try the vertical
        return _tValueForPointOnCubicCurve(point, (pt1, pt2, pt3, pt4), isHorizontal=1)
    if len(solutions) > 1:
        intersectionLenghts = {}
        for t in solutions:
            tp = _getCubicPoint(t, pt1, pt2, pt3, pt4)
            dist = _distance(tp, point)
            intersectionLenghts[dist] = t
        minDist = min(intersectionLenghts.keys())
        solutions = [intersectionLenghts[minDist]]
    return solutions