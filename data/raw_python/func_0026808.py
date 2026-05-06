def _prepPointsForSegments(points):
    """
    Move any off curves at the end of the contour
    to the beginning of the contour. This makes
    segmentation easier.
    """
    while 1:
        point = points[-1]
        if point.segmentType:
            break
        else:
            point = points.pop()
            points.insert(0, point)
            continue
        break