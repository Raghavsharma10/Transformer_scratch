def theta_limit(theta):
    """
    Angle theta is periodic with period pi.
    Constrain theta such that -pi/2<theta<=pi/2.

    Parameters
    ----------
    theta : float
        Input angle.

    Returns
    -------
    theta : float
        Rotate angle.
    """
    while theta <= -1 * np.pi / 2:
        theta += np.pi
    while theta > np.pi / 2:
        theta -= np.pi
    return theta