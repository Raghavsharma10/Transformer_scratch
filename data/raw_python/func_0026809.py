def _reversePoints(points):
    """
    Reverse the points. This differs from the
    reversal point pen in RoboFab in that it doesn't
    worry about maintaing the start point position.
    That has no benefit within the context of this module.
    """
    # copy the points
    points = _copyPoints(points)
    # find the first on curve type and recycle
    # it for the last on curve type
    firstOnCurve = None
    for index, point in enumerate(points):
        if point.segmentType is not None:
            firstOnCurve = index
            break
    lastSegmentType = points[firstOnCurve].segmentType
    # reverse the points
    points = reversed(points)
    # work through the reversed remaining points
    final = []
    for point in points:
        segmentType = point.segmentType
        if segmentType is not None:
            point.segmentType = lastSegmentType
            lastSegmentType = segmentType
        final.append(point)
    # move any offcurves at the end of the points
    # to the start of the points
    _prepPointsForSegments(final)
    # done
    return final