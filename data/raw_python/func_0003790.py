def quaternion_product(quat1, quat2):
    """Return the quaternion product of the two arguments"""
    return np.array([
        quat1[0]*quat2[0] - np.dot(quat1[1:], quat2[1:]),
        quat1[0]*quat2[1] + quat2[0]*quat1[1] + quat1[2]*quat2[3] - quat1[3]*quat2[2],
        quat1[0]*quat2[2] + quat2[0]*quat1[2] + quat1[3]*quat2[1] - quat1[1]*quat2[3],
        quat1[0]*quat2[3] + quat2[0]*quat1[3] + quat1[1]*quat2[2] - quat1[2]*quat2[1]
    ], float)