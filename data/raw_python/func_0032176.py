def rotateAroundVector(v1, w, theta_deg):
    """Rotate vector v1 by an angle theta around w

    Taken from https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    (see Section "Rotating a vector")

    Notes:
    Rotating the x axis 90 degrees about the y axis gives -z
    Rotating the x axis 90 degrees about the z axis gives +y
    """

    ct = np.cos(np.radians(theta_deg))
    st = np.sin(np.radians(theta_deg))
    term1 = v1*ct
    term2 = np.cross(w, v1) * st
    term3 = np.dot(w, v1)
    term3 = w * term3 * (1-ct)

    return term1 + term2 + term3