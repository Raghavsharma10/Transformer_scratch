def azimuth(poly):
    """Azimuth of a polygon poly"""
    num = len(poly) - 1
    vec = unit_normal(poly[0], poly[1], poly[num])
    vec_azi = np.array([vec[0], vec[1], 0])
    vec_n = np.array([0, 1, 0])
    # update by Santosh
    # angle2vecs gives the smallest angle between the vectors
    # so for a west wall angle2vecs will give 90
    # the following 'if' statement will make sure 270 is returned
    x_vector = vec_azi[0]
    if x_vector < 0:
        return 360 - angle2vecs(vec_azi, vec_n)
    else:
        return angle2vecs(vec_azi, vec_n)