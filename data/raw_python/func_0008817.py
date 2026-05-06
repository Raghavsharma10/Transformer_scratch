def galactic2fk5(l, b):
    """
    Convert galactic l/b to fk5 ra/dec

    Parameters
    ----------
    l, b : float
        Galactic coordinates in radians.

    Returns
    -------
    ra, dec : float
        FK5 ecliptic coordinates in radians.
    """
    a = SkyCoord(l, b, unit=(u.radian, u.radian), frame='galactic')
    return a.fk5.ra.radian, a.fk5.dec.radian