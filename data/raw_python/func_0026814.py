def _flattenSegment(segment, approximateSegmentLength=_approximateSegmentLength):
    """
    Flatten the curve segment int a list of points.
    The first and last points in the segment must be
    on curves. The returned list of points will not
    include the first on curve point.
    false curves (where the off curves are not any
    different from the on curves) must not be sent here.
    duplicate points must not be sent here.
    """
    onCurve1, offCurve1, offCurve2, onCurve2 = segment
    if _pointOnLine(onCurve1, onCurve2, offCurve1) and _pointOnLine(onCurve1, onCurve2, offCurve2):
        return [onCurve2]
    est = _estimateCubicCurveLength(onCurve1, offCurve1, offCurve2, onCurve2) / approximateSegmentLength
    flat = []
    minStep = 0.1564
    step = 1.0 / est
    if step > .3:
        step = minStep
    t = step
    while t < 1:
        pt = _getCubicPoint(t, onCurve1, offCurve1, offCurve2, onCurve2)
        # ignore when point is in the same direction as the on - off curve line
        if not _pointOnLine(offCurve2, onCurve2, pt) and not _pointOnLine(onCurve1, offCurve1, pt):
            flat.append(pt)
        t += step
    flat.append(onCurve2)
    return flat