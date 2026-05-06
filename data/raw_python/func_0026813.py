def _scaleSinglePoint(point, scale=1, convertToInteger=True):
    """
    Scale a single point
    """
    x, y = point
    if convertToInteger:
        return int(round(x * scale)), int(round(y * scale))
    else:
        return (x * scale, y * scale)