def declinationRotationMatrix(theta_deg):
    """Construct the rotation matrix for a rotation of the declination
    coords (i.e around the axis of ra=90, dec=0)

    Taken from Section 3.3 of Arfken and Weber (Eqn 3.91)
    Modfied the signs of the sines so that a rotation of the zenith
    vector by 90 degrees ends up at ra, dec = 0,0
    """

    ct = np.cos(np.radians(theta_deg))
    st = np.sin(np.radians(theta_deg))

    mat = np.zeros((3,3))

    mat[0,0] = ct
    mat[0,2] = -st
    mat[1,1] = 1
    mat[2,0] = st
    mat[2,2] = ct

    return mat