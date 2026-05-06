def create_plane(width=1, height=1, width_segments=1, height_segments=1,
                 direction='+z'):
    """ Generate vertices & indices for a filled and outlined plane.

    Parameters
    ----------
    width : float
        Plane width.
    height : float
        Plane height.
    width_segments : int
        Plane segments count along the width.
    height_segments : float
        Plane segments count along the height.
    direction: unicode
        ``{'-x', '+x', '-y', '+y', '-z', '+z'}``
        Direction the plane will be facing.

    Returns
    -------
    vertices : array
        Array of vertices suitable for use as a VertexBuffer.
    faces : array
        Indices to use to produce a filled plane.
    outline : array
        Indices to use to produce an outline of the plane.

    References
    ----------
    .. [1] Cabello, R. (n.d.). PlaneBufferGeometry.js. Retrieved May 12, 2015,
        from http://git.io/vU1Fh
    """

    x_grid = width_segments
    y_grid = height_segments

    x_grid1 = x_grid + 1
    y_grid1 = y_grid + 1

    # Positions, normals and texcoords.
    positions = np.zeros(x_grid1 * y_grid1 * 3)
    normals = np.zeros(x_grid1 * y_grid1 * 3)
    texcoords = np.zeros(x_grid1 * y_grid1 * 2)

    y = np.arange(y_grid1) * height / y_grid - height / 2
    x = np.arange(x_grid1) * width / x_grid - width / 2

    positions[::3] = np.tile(x, y_grid1)
    positions[1::3] = -np.repeat(y, x_grid1)

    normals[2::3] = 1

    texcoords[::2] = np.tile(np.arange(x_grid1) / x_grid, y_grid1)
    texcoords[1::2] = np.repeat(1 - np.arange(y_grid1) / y_grid, x_grid1)

    # Faces and outline.
    faces, outline = [], []
    for i_y in range(y_grid):
        for i_x in range(x_grid):
            a = i_x + x_grid1 * i_y
            b = i_x + x_grid1 * (i_y + 1)
            c = (i_x + 1) + x_grid1 * (i_y + 1)
            d = (i_x + 1) + x_grid1 * i_y

            faces.extend(((a, b, d), (b, c, d)))
            outline.extend(((a, b), (b, c), (c, d), (d, a)))

    positions = np.reshape(positions, (-1, 3))
    texcoords = np.reshape(texcoords, (-1, 2))
    normals = np.reshape(normals, (-1, 3))

    faces = np.reshape(faces, (-1, 3)).astype(np.uint32)
    outline = np.reshape(outline, (-1, 2)).astype(np.uint32)

    direction = direction.lower()
    if direction in ('-x', '+x'):
        shift, neutral_axis = 1, 0
    elif direction in ('-y', '+y'):
        shift, neutral_axis = -1, 1
    elif direction in ('-z', '+z'):
        shift, neutral_axis = 0, 2

    sign = -1 if '-' in direction else 1

    positions = np.roll(positions, shift, -1)
    normals = np.roll(normals, shift, -1) * sign
    colors = np.ravel(positions)
    colors = np.hstack((np.reshape(np.interp(colors,
                                             (np.min(colors),
                                              np.max(colors)),
                                             (0, 1)),
                                   positions.shape),
                        np.ones((positions.shape[0], 1))))
    colors[..., neutral_axis] = 0

    vertices = np.zeros(positions.shape[0],
                        [('position', np.float32, 3),
                         ('texcoord', np.float32, 2),
                         ('normal', np.float32, 3),
                         ('color', np.float32, 4)])

    vertices['position'] = positions
    vertices['texcoord'] = texcoords
    vertices['normal'] = normals
    vertices['color'] = colors

    return vertices, faces, outline