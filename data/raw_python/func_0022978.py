def create_cone(cols, radius=1.0, length=1.0):
    """Create a cone

    Parameters
    ----------
    cols : int
        Number of faces.
    radius : float
        Base cone radius.
    length : float
        Length of the cone.

    Returns
    -------
    cone : MeshData
        Vertices and faces computed for a cone surface.
    """
    verts = np.empty((cols+1, 3), dtype=np.float32)
    # compute vertexes
    th = np.linspace(2 * np.pi, 0, cols+1).reshape(1, cols+1)
    verts[:-1, 2] = 0.0
    verts[:-1, 0] = radius * np.cos(th[0, :-1])  # x = r cos(th)
    verts[:-1, 1] = radius * np.sin(th[0, :-1])  # y = r sin(th)
    # Add the extremity
    verts[-1, 0] = 0.0
    verts[-1, 1] = 0.0
    verts[-1, 2] = length
    verts = verts.reshape((cols+1), 3)  # just reshape: no redundant vertices
    # compute faces
    faces = np.empty((cols, 3), dtype=np.uint32)
    template = np.array([[0, 1]])
    for pos in range(cols):
        faces[pos, :-1] = template + pos
    faces[:, 2] = cols
    faces[-1, 1] = 0

    return MeshData(vertices=verts, faces=faces)