def quaternion_rotation(quat, vector):
    """Apply the rotation represented by the quaternion to the vector

       Warning: This only works correctly for normalized quaternions.
    """
    dp = np.dot(quat[1:], vector)
    cos = (2*quat[0]*quat[0] - 1)
    return np.array([
        2 * (quat[0] * (quat[2] * vector[2] - quat[3] * vector[1]) + quat[1] * dp) + cos * vector[0],
        2 * (quat[0] * (quat[3] * vector[0] - quat[1] * vector[2]) + quat[2] * dp) + cos * vector[1],
        2 * (quat[0] * (quat[1] * vector[1] - quat[2] * vector[0]) + quat[3] * dp) + cos * vector[2]
    ], float)