def normals(vertices, indices):
    """
    Compute normals over a triangulated surface

    Parameters
    ----------

    vertices : ndarray (n,3)
        triangles vertices

    indices : ndarray (p,3)
        triangles indices
    """

    # Compact similar vertices
    vertices, indices, mapping = compact(vertices, indices)

    T = vertices[indices]
    N = np.cross(T[:, 1] - T[:, 0], T[:, 2]-T[:, 0])
    L = np.sqrt(np.sum(N * N, axis=1))
    L[L == 0] = 1.0  # prevent divide-by-zero
    N /= L[:, np.newaxis]
    normals = np.zeros_like(vertices)
    normals[indices[:, 0]] += N
    normals[indices[:, 1]] += N
    normals[indices[:, 2]] += N
    L = np.sqrt(np.sum(normals*normals, axis=1))
    L[L == 0] = 1.0
    normals /= L[:, np.newaxis]

    return normals[mapping]