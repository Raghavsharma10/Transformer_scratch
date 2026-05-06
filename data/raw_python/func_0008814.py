def dist_rhumb(ra1, dec1, ra2, dec2):
    """
    Calculate the Rhumb line distance between two points [1]_.
    A Rhumb line between two points is one which follows a constant bearing.

    Parameters
    ----------
    ra1, dec1, ra2, dec2 : float
        The position of the two points (degrees).

    Returns
    -------
    dist : float
        The distance between the two points along a line of constant bearing.

    Notes
    -----
    .. [1] `Rhumb line <https://en.wikipedia.org/wiki/Rhumb_line>`_
    """
    # verified against website to give correct results
    phi1 = np.radians(dec1)
    phi2 = np.radians(dec2)
    dphi = phi2 - phi1
    lambda1 = np.radians(ra1)
    lambda2 = np.radians(ra2)
    dpsi = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    if dpsi < 1e-12:
        q = np.cos(phi1)
    else:
        q = dpsi / dphi
    dlambda = lambda2 - lambda1
    if dlambda > np.pi:
        dlambda -= 2 * np.pi
    dist = np.hypot(dphi, q * dlambda)
    return np.degrees(dist)