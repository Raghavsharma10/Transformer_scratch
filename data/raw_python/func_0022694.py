def rotate(angle, axis, dtype=None):
    """The 3x3 rotation matrix for rotation about a vector.

    Parameters
    ----------
    angle : float
        The angle of rotation, in degrees.
    axis : ndarray
        The x, y, z coordinates of the axis direction vector.
    """
    angle = np.radians(angle)
    assert len(axis) == 3
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    M = np.array([[cx * x + c, cy * x - z * s, cz * x + y * s, .0],
                  [cx * y + z * s, cy * y + c, cz * y - x * s, 0.],
                  [cx * z - y * s, cy * z + x * s, cz * z + c, 0.],
                  [0., 0., 0., 1.]], dtype).T
    return M