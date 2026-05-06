def height(poly):
    """height"""
    num = len(poly)
    hgt = 0.0
    for i in range(num):
        hgt += (poly[i][2])
    return hgt/num