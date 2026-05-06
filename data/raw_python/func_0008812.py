def bear(ra1, dec1, ra2, dec2):
    """
    Calculate the bearing of point 2 from point 1 along a great circle.
    The bearing is East of North and is in [0, 360), whereas position angle is also East of North but (-180,180]

    Parameters
    ----------
    ra1, dec1, ra2, dec2 : float
        The sky coordinates (degrees) of the two points.

    Returns
    -------
    bear : float
        The bearing of point 2 from point 1 (degrees).
    """
    rdec1 = np.radians(dec1)
    rdec2 = np.radians(dec2)
    rdlon = np.radians(ra2-ra1)
    y = np.sin(rdlon) * np.cos(rdec2)
    x = np.cos(rdec1) * np.sin(rdec2)
    x -= np.sin(rdec1) * np.cos(rdec2) * np.cos(rdlon)
    return np.degrees(np.arctan2(y, x))