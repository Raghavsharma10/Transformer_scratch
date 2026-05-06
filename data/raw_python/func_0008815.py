def bear_rhumb(ra1, dec1, ra2, dec2):
    """
    Calculate the bearing of point 2 from point 1 along a Rhumb line.
    The bearing is East of North and is in [0, 360), whereas position angle is also East of North but (-180,180]

    Parameters
    ----------
    ra1, dec1, ra2, dec2 : float
        The sky coordinates (degrees) of the two points.

    Returns
    -------
    dist : float
        The bearing of point 2 from point 1 along a Rhumb line (degrees).
    """
    # verified against website to give correct results
    phi1 = np.radians(dec1)
    phi2 = np.radians(dec2)
    lambda1 = np.radians(ra1)
    lambda2 = np.radians(ra2)
    dlambda = lambda2 - lambda1

    dpsi = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))

    theta = np.arctan2(dlambda, dpsi)
    return np.degrees(theta)