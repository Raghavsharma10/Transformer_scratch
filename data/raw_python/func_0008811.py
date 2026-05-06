def gcd(ra1, dec1, ra2, dec2):
    """
    Calculate the great circle distance between to points using the haversine formula [1]_.


    Parameters
    ----------
    ra1, dec1, ra2, dec2 : float
        The coordinates of the two points of interest.
        Units are in degrees.

    Returns
    -------
    dist : float
        The distance between the two points in degrees.

    Notes
    -----
    This duplicates the functionality of astropy but is faster as there is no creation of SkyCoords objects.

    .. [1] `Haversine formula <https://en.wikipedia.org/wiki/Haversine_formula>`_
    """
    # TODO:  Vincenty formula see - https://en.wikipedia.org/wiki/Great-circle_distance
    dlon = ra2 - ra1
    dlat = dec2 - dec1
    a = np.sin(np.radians(dlat) / 2) ** 2
    a += np.cos(np.radians(dec1)) * np.cos(np.radians(dec2)) * np.sin(np.radians(dlon) / 2) ** 2
    sep = np.degrees(2 * np.arcsin(np.minimum(1, np.sqrt(a))))
    return sep