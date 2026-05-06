def vol_tehrahedron(poly):
    """volume of a irregular tetrahedron"""
    a_pnt = np.array(poly[0])
    b_pnt = np.array(poly[1])
    c_pnt = np.array(poly[2])
    d_pnt = np.array(poly[3])
    return abs(np.dot(
        (a_pnt-d_pnt), np.cross((b_pnt-d_pnt), (c_pnt-d_pnt))) / 6)