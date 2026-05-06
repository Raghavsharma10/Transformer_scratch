def _estimateCubicCurveLength(pt0, pt1, pt2, pt3, precision=10):
    """
    Estimate the length of this curve by iterating
    through it and averaging the length of the flat bits.
    """
    points = []
    length = 0
    step = 1.0 / precision
    factors = range(0, precision + 1)
    for i in factors:
        points.append(_getCubicPoint(i * step, pt0, pt1, pt2, pt3))
    for i in range(len(points) - 1):
        pta = points[i]
        ptb = points[i + 1]
        length += _distance(pta, ptb)
    return length