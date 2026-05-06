def translate_rhumb(ra, dec, r, theta):
    """
    Translate a given point a distance r in the (initial) direction theta, along a Rhumb line.

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
    # verified against website to give correct results
    # with the help of http://williams.best.vwh.net/avform.htm#Rhumb
    delta = np.radians(r)
    phi1 = np.radians(dec)
    phi2 = phi1 + delta * np.cos(np.radians(theta))
    dphi = phi2 - phi1

    if abs(dphi) < 1e-9:
        q = np.cos(phi1)
    else:
        dpsi = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
        q = dphi / dpsi

    lambda1 = np.radians(ra)
    dlambda = delta * np.sin(np.radians(theta)) / q
    lambda2 = lambda1 + dlambda

    ra_out = np.degrees(lambda2)
    dec_out = np.degrees(phi2)
    return ra_out, dec_out