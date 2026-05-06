def rotateInDeclination(v1, theta_deg):
    """Rotation is chosen so a rotation of 90 degrees from zenith
    ends up at ra=0, dec=0"""
    axis = np.array([0,-1,0])
    return rotateAroundVector(v1, axis, theta_deg)