def triangle_normal(a, b, c):
    """Return a vector orthogonal to the given triangle

       Arguments:
         a, b, c  --  three 3D numpy vectors
    """
    normal = np.cross(a - c, b - c)
    norm = np.linalg.norm(normal)
    return normal/norm