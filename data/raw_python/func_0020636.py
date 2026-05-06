def surf_keep_cortex(surf, cortex):
    """
    Remove medial wall from cortical surface to ensure that shortest paths are only calculated through the cortex.

    Inputs
    -------
    surf : Tuple containing two numpy arrays of shape (n_nodes,3). Each node of the first array specifies the x, y, z
           coordinates one node of the surface mesh. Each node of the second array specifies the indices of the three
           nodes building one triangle of the surface mesh.
           (e.g. the output from nibabel.freesurfer.io.read_geometry)
    cortex : Array with indices of vertices included in within the cortex.
             (e.g. the output from nibabel.freesurfer.io.read_label)
    """

    # split surface into vertices and triangles
    vertices, triangles = surf

    # keep only the vertices within the cortex label
    cortex_vertices = np.array(vertices[cortex], dtype=np.float64)

    # keep only the triangles within the cortex label
    cortex_triangles = triangles_keep_cortex(triangles, cortex)

    return cortex_vertices, cortex_triangles