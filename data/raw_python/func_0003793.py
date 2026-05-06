def quaternion_to_rotation_matrix(quaternion):
    """Compute the rotation matrix representated by the quaternion"""
    c, x, y, z = quaternion
    return np.array([
        [c*c + x*x - y*y - z*z, 2*x*y - 2*c*z,         2*x*z + 2*c*y        ],
        [2*x*y + 2*c*z,         c*c - x*x + y*y - z*z, 2*y*z - 2*c*x        ],
        [2*x*z - 2*c*y,         2*y*z + 2*c*x,         c*c - x*x - y*y + z*z]
    ], float)