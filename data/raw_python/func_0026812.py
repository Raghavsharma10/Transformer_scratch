def _scalePoints(points, scale=1, convertToInteger=True):
    """
    Scale points and optionally convert them to integers.
    """
    if convertToInteger:
        points = [
            (int(round(x * scale)), int(round(y * scale)))
            for (x, y) in points
        ]
    else:
        points = [(x * scale, y * scale) for (x, y) in points]
    return points