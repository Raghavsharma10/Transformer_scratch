def vol_tehrahedron(poly):
    """volume of a irregular tetrahedron"""
    p_a = np.array(poly[0])
    p_b = np.array(poly[1])
    p_c = np.array(poly[2])
    p_d = np.array(poly[3])
    return abs(np.dot(
        np.subtract(p_a, p_d),
        np.cross(
            np.subtract(p_b, p_d),
            np.subtract(p_c, p_d))) / 6)