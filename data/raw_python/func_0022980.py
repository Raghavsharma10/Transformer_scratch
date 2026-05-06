def create_grid_mesh(xs, ys, zs):
    '''Generate vertices and indices for an implicitly connected mesh.

    The intention is that this makes it simple to generate a mesh
    from meshgrid data.

    Parameters
    ----------
    xs : ndarray
        A 2d array of x coordinates for the vertices of the mesh. Must
        have the same dimensions as ys and zs.
    ys : ndarray
        A 2d array of y coordinates for the vertices of the mesh. Must
        have the same dimensions as xs and zs.
    zs : ndarray
        A 2d array of z coordinates for the vertices of the mesh. Must
        have the same dimensions as xs and ys.

    Returns
    -------
    vertices : ndarray
        The array of vertices in the mesh.
    indices : ndarray
        The array of indices for the mesh.
    '''

    shape = xs.shape
    length = shape[0] * shape[1]

    vertices = np.zeros((length, 3))

    vertices[:, 0] = xs.reshape(length)
    vertices[:, 1] = ys.reshape(length)
    vertices[:, 2] = zs.reshape(length)

    basic_indices = np.array([0, 1, 1 + shape[1], 0,
                              0 + shape[1], 1 + shape[1]],
                             dtype=np.uint32)

    inner_grid_length = (shape[0] - 1) * (shape[1] - 1)

    offsets = np.arange(inner_grid_length)
    offsets += np.repeat(np.arange(shape[0] - 1), shape[1] - 1)
    offsets = np.repeat(offsets, 6)
    indices = np.resize(basic_indices, len(offsets)) + offsets

    indices = indices.reshape((len(indices) // 3, 3))

    return vertices, indices