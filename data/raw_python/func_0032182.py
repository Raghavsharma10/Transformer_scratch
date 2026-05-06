def rotateInZMat(theta_deg):
    """Rotate a vector theta degrees around the z-axis

    Equivalent to yaw left

    Rotates the vector in the sense that the x-axis is rotated
    towards the y-axis. If looking along the z-axis (which is
    not the way you usually look at it), the vector rotates
    clockwise.

    If sitting on the vector [1,0,0], the rotation is towards the left

    Input:
    theta_deg   (float) Angle through which vectors should be
                rotated in degrees

    Returns:
    A matrix

    To rotate a vector, premultiply by this matrix.
    To rotate the coord sys underneath the vector, post multiply

    """

    ct = np.cos( np.radians(theta_deg))
    st = np.sin( np.radians(theta_deg))
    rMat = np.array([  [ ct, -st, 0],
                       [ st,  ct, 0],
                       [  0,   0, 1],
                    ])

    return rMat