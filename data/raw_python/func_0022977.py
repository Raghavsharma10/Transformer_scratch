def create_cylinder(rows, cols, radius=[1.0, 1.0], length=1.0, offset=False):
    """Create a cylinder

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    radius : tuple of float
        Cylinder radii.
    length : float
        Length of the cylinder.
    offset : bool
        Rotate each row by half a column.

    Returns
    -------
    cylinder : MeshData
        Vertices and faces computed for a cylindrical surface.
    """
    verts = np.empty((rows+1, cols, 3), dtype=np.float32)
    if isinstance(radius, int):
        radius = [radius, radius]  # convert to list
    # compute vertices
    th = np.linspace(2 * np.pi, 0, cols).reshape(1, cols)
    # radius as a function of z
    r = np.linspace(radius[0], radius[1], num=rows+1,
                    endpoint=True).reshape(rows+1, 1)
    verts[..., 2] = np.linspace(0, length, num=rows+1,
                                endpoint=True).reshape(rows+1, 1)  # z
    if offset:
        # rotate each row by 1/2 column
        th = th + ((np.pi / cols) * np.arange(rows+1).reshape(rows+1, 1))
    verts[..., 0] = r * np.cos(th)  # x = r cos(th)
    verts[..., 1] = r * np.sin(th)  # y = r sin(th)
    # just reshape: no redundant vertices...
    verts = verts.reshape((rows+1)*cols, 3)
    # compute faces
    faces = np.empty((rows*cols*2, 3), dtype=np.uint32)
    rowtemplate1 = (((np.arange(cols).reshape(cols, 1) +
                      np.array([[0, 1, 0]])) % cols) +
                    np.array([[0, 0, cols]]))
    rowtemplate2 = (((np.arange(cols).reshape(cols, 1) +
                      np.array([[0, 1, 1]])) % cols) +
                    np.array([[cols, 0, cols]]))
    for row in range(rows):
        start = row * cols * 2
        faces[start:start+cols] = rowtemplate1 + row * cols
        faces[start+cols:start+(cols*2)] = rowtemplate2 + row * cols

    return MeshData(vertices=verts, faces=faces)