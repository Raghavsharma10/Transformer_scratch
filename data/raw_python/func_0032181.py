def rotateAboutVectorMatrix(vec, theta_deg):
    """Construct the matrix that rotates vector a about
    vector vec by an angle of theta_deg degrees

    Taken from
    http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

    Input:
    theta_deg   (float) Angle through which vectors should be
                rotated in degrees

    Returns:
    A matrix

    To rotate a vector, premultiply by this matrix.
    To rotate the coord sys underneath the vector, post multiply

    """
    ct = np.cos(np.radians(theta_deg))
    st = np.sin(np.radians(theta_deg))

    # Ensure vector has normal length
    vec /= np.linalg.norm(vec)
    assert( np.all( np.isfinite(vec)))

    # compute the three terms
    term1 = ct * np.eye(3)

    ucross = np.zeros( (3,3))
    ucross[0] = [0, -vec[2], vec[1]]
    ucross[1] = [vec[2], 0, -vec[0]]
    ucross[2] = [-vec[1], vec[0], 0]

    term2 = st*ucross

    ufunny = np.zeros( (3,3))
    for i in range(0,3):
        for j in range(i,3):
            ufunny[i,j] = vec[i]*vec[j]
            ufunny[j,i] = ufunny[i,j]

    term3 = (1-ct) * ufunny

    return term1 + term2 + term3