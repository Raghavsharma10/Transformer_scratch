def tilt(poly):
    """Tilt of a polygon poly"""
    num = len(poly) - 1
    vec = unit_normal(poly[0], poly[1], poly[num])
    vec_alt = np.array([vec[0], vec[1], vec[2]])
    vec_z = np.array([0, 0, 1])
    # return (90 - angle2vecs(vec_alt, vec_z)) # update by Santosh
    return angle2vecs(vec_alt, vec_z)