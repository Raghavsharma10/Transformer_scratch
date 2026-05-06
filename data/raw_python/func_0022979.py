def create_arrow(rows, cols, radius=0.1, length=1.0,
                 cone_radius=None, cone_length=None):
    """Create a 3D arrow using a cylinder plus cone

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    radius : float
        Base cylinder radius.
    length : float
        Length of the arrow.
    cone_radius : float
        Radius of the cone base.
           If None, then this defaults to 2x the cylinder radius.
    cone_length : float
        Length of the cone.
           If None, then this defaults to 1/3 of the arrow length.

    Returns
    -------
    arrow : MeshData
        Vertices and faces computed for a cone surface.
    """
    # create the cylinder
    md_cyl = None
    if cone_radius is None:
        cone_radius = radius*2.0
    if cone_length is None:
        con_L = length/3.0
        cyl_L = length*2.0/3.0
    else:
        cyl_L = max(0, length - cone_length)
        con_L = min(cone_length, length)
    if cyl_L != 0:
        md_cyl = create_cylinder(rows, cols, radius=[radius, radius],
                                 length=cyl_L)
    # create the cone
    md_con = create_cone(cols, radius=cone_radius, length=con_L)
    verts = md_con.get_vertices()
    nbr_verts_con = verts.size//3
    faces = md_con.get_faces()
    if md_cyl is not None:
        trans = np.array([[0.0, 0.0, cyl_L]])
        verts = np.vstack((verts+trans, md_cyl.get_vertices()))
        faces = np.vstack((faces, md_cyl.get_faces()+nbr_verts_con))

    return MeshData(vertices=verts, faces=faces)