def height(poly):
    """Height of a polygon poly"""
    num = len(poly) - 1
    if abs(poly[num][2] - poly[0][2]) > abs(poly[1][2] - poly[0][2]):
        return dist(poly[num], poly[0])
    elif abs(poly[num][2] - poly[0][2]) < abs(poly[1][2] - poly[0][2]):
        return dist(poly[1], poly[0])
    else:
        return min(dist(poly[num], poly[0]), dist(poly[1], poly[0]))