def gc2gdlat(gclat):
    """Converts geocentric latitude to geodetic latitude using WGS84.

    Parameters
    ==========
    gclat : array_like
        Geocentric latitude

    Returns
    =======
    gdlat : ndarray or float
        Geodetic latitude

    """
    WGS84_e2 = 0.006694379990141317  # WGS84 first eccentricity squared
    return np.rad2deg(-np.arctan(np.tan(np.deg2rad(gclat))/(WGS84_e2 - 1)))