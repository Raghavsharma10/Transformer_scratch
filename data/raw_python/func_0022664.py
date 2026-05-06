def iso_mesh_line(vertices, tris, vertex_data, levels):
    """Generate an isocurve from vertex data in a surface mesh.

    Parameters
    ----------
    vertices : ndarray, shape (Nv, 3)
        Vertex coordinates.
    tris : ndarray, shape (Nf, 3)
        Indices of triangular element into the vertices array.
    vertex_data : ndarray, shape (Nv,)
        data at vertex.
    levels : ndarray, shape (Nl,)
        Levels at which to generate an isocurve

    Returns
    -------
    lines : ndarray, shape (Nvout, 3)
        Vertex coordinates for lines points
    connects : ndarray, shape (Ne, 2)
        Indices of line element into the vertex array.
    vertex_level: ndarray, shape (Nvout,)
        level for vertex in lines

    Notes
    -----
    Uses a marching squares algorithm to generate the isolines.
    """

    lines = None
    connects = None
    vertex_level = None
    level_index = None
    if not all([isinstance(x, np.ndarray) for x in (vertices, tris,
                vertex_data, levels)]):
        raise ValueError('all inputs must be numpy arrays')
    if vertices.shape[1] <= 3:
        verts = vertices
    elif vertices.shape[1] == 4:
        verts = vertices[:, :-1]
    else:
        verts = None
    if (verts is not None and tris.shape[1] == 3 and
            vertex_data.shape[0] == verts.shape[0]):
        edges = np.vstack((tris.reshape((-1)),
                           np.roll(tris, -1, axis=1).reshape((-1)))).T
        edge_datas = vertex_data[edges]
        edge_coors = verts[edges].reshape(tris.shape[0]*3, 2, 3)
        for lev in levels:
            # index for select edges with vertices have only False - True
            # or True - False at extremity
            index = (edge_datas >= lev)
            index = index[:, 0] ^ index[:, 1]  # xor calculation
            # Selectect edge
            edge_datas_Ok = edge_datas[index, :]
            xyz = edge_coors[index]
            # Linear interpolation
            ratio = np.array([(lev - edge_datas_Ok[:, 0]) /
                              (edge_datas_Ok[:, 1] - edge_datas_Ok[:, 0])])
            point = xyz[:, 0, :] + ratio.T * (xyz[:, 1, :] - xyz[:, 0, :])
            nbr = point.shape[0]//2
            if connects is not None:
                connect = np.arange(0, nbr*2).reshape((nbr, 2)) + \
                    len(lines)
                connects = np.append(connects, connect, axis=0)
                lines = np.append(lines, point, axis=0)
                vertex_level = np.append(vertex_level,
                                         np.zeros(len(point)) +
                                         lev)
                level_index = np.append(level_index, np.array(len(point)))
            else:
                lines = point
                connects = np.arange(0, nbr*2).reshape((nbr, 2))
                vertex_level = np.zeros(len(point)) + lev
                level_index = np.array(len(point))

            vertex_level = vertex_level.reshape((vertex_level.size, 1))

    return lines, connects, vertex_level, level_index