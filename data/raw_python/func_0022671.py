def triangulate(vertices):
    """Triangulate a set of vertices

    Parameters
    ----------
    vertices : array-like
        The vertices.

    Returns
    -------
    vertices : array-like
        The vertices.
    tringles : array-like
        The triangles.
    """
    n = len(vertices)
    vertices = np.asarray(vertices)
    zmean = vertices[:, 2].mean()
    vertices_2d = vertices[:, :2]
    segments = np.repeat(np.arange(n + 1), 2)[1:-1]
    segments[-2:] = n - 1, 0

    if _TRIANGLE_AVAILABLE:
        vertices_2d, triangles = _triangulate_cpp(vertices_2d, segments)
    else:
        vertices_2d, triangles = _triangulate_python(vertices_2d, segments)

    vertices = np.empty((len(vertices_2d), 3))
    vertices[:, :2] = vertices_2d
    vertices[:, 2] = zmean
    return vertices, triangles