def translate(ra, dec, r, theta):
    """
    Translate a given point a distance r in the (initial) direction theta, along a  great circle.


    Parameters
    ----------
    ra, dec : float
        The initial point of interest (degrees).
    r, theta : float
        The distance and initial direction to translate (degrees).

    Returns
    -------
    ra, dec : float
        The translated position (degrees).
    """
    factor = np.sin(np.radians(dec)) * np.cos(np.radians(r))
    factor += np.cos(np.radians(dec)) * np.sin(np.radians(r)) * np.cos(np.radians(theta))
    dec_out = np.degrees(np.arcsin(factor))

    y = np.sin(np.radians(theta)) * np.sin(np.radians(r)) * np.cos(np.radians(dec))
    x = np.cos(np.radians(r)) - np.sin(np.radians(dec)) * np.sin(np.radians(dec_out))
    ra_out = ra + np.degrees(np.arctan2(y, x))
    return ra_out, dec_out