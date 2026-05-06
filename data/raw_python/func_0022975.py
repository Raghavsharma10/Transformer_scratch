def create_box(width=1, height=1, depth=1, width_segments=1, height_segments=1,
               depth_segments=1, planes=None):
    """ Generate vertices & indices for a filled and outlined box.

    Parameters
    ----------
    width : float
        Box width.
    height : float
        Box height.
    depth : float
        Box depth.
    width_segments : int
        Box segments count along the width.
    height_segments : float
        Box segments count along the height.
    depth_segments : float
        Box segments count along the depth.
    planes: array_like
        Any combination of ``{'-x', '+x', '-y', '+y', '-z', '+z'}``
        Included planes in the box construction.

    Returns
    -------
    vertices : array
        Array of vertices suitable for use as a VertexBuffer.
    faces : array
        Indices to use to produce a filled box.
    outline : array
        Indices to use to produce an outline of the box.
    """

    planes = (('+x', '-x', '+y', '-y', '+z', '-z')
              if planes is None else
              [d.lower() for d in planes])

    w_s, h_s, d_s = width_segments, height_segments, depth_segments

    planes_m = []
    if '-z' in planes:
        planes_m.append(create_plane(width, depth, w_s, d_s, '-z'))
        planes_m[-1][0]['position'][..., 2] -= height / 2
    if '+z' in planes:
        planes_m.append(create_plane(width, depth, w_s, d_s, '+z'))
        planes_m[-1][0]['position'][..., 2] += height / 2

    if '-y' in planes:
        planes_m.append(create_plane(height, width, h_s, w_s, '-y'))
        planes_m[-1][0]['position'][..., 1] -= depth / 2
    if '+y' in planes:
        planes_m.append(create_plane(height, width, h_s, w_s, '+y'))
        planes_m[-1][0]['position'][..., 1] += depth / 2

    if '-x' in planes:
        planes_m.append(create_plane(depth, height, d_s, h_s, '-x'))
        planes_m[-1][0]['position'][..., 0] -= width / 2
    if '+x' in planes:
        planes_m.append(create_plane(depth, height, d_s, h_s, '+x'))
        planes_m[-1][0]['position'][..., 0] += width / 2

    positions = np.zeros((0, 3), dtype=np.float32)
    texcoords = np.zeros((0, 2), dtype=np.float32)
    normals = np.zeros((0, 3), dtype=np.float32)

    faces = np.zeros((0, 3), dtype=np.uint32)
    outline = np.zeros((0, 2), dtype=np.uint32)

    offset = 0
    for vertices_p, faces_p, outline_p in planes_m:
        positions = np.vstack((positions, vertices_p['position']))
        texcoords = np.vstack((texcoords, vertices_p['texcoord']))
        normals = np.vstack((normals, vertices_p['normal']))

        faces = np.vstack((faces, faces_p + offset))
        outline = np.vstack((outline, outline_p + offset))
        offset += vertices_p['position'].shape[0]

    vertices = np.zeros(positions.shape[0],
                        [('position', np.float32, 3),
                         ('texcoord', np.float32, 2),
                         ('normal', np.float32, 3),
                         ('color', np.float32, 4)])

    colors = np.ravel(positions)
    colors = np.hstack((np.reshape(np.interp(colors,
                                             (np.min(colors),
                                              np.max(colors)),
                                             (0, 1)),
                                   positions.shape),
                        np.ones((positions.shape[0], 1))))

    vertices['position'] = positions
    vertices['texcoord'] = texcoords
    vertices['normal'] = normals
    vertices['color'] = colors

    return vertices, faces, outline