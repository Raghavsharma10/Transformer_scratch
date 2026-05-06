def getcoords(ddtt):
    """return the coordinates of the surface"""
    n_vertices_index = ddtt.objls.index('Number_of_Vertices')
    first_x = n_vertices_index + 1 # X of first coordinate
    pts = ddtt.obj[first_x:]
    return list(grouper(3, pts))