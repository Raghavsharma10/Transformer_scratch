def sky_dist(src1, src2):
    """
    Great circle distance between two sources.
    A check is made to determine if the two sources are the same object, in this case
    the distance is zero.

    Parameters
    ----------
    src1, src2 : object
        Two sources to check. Objects must have parameters (ra,dec) in degrees.

    Returns
    -------
    distance : float
        The distance between the two sources.

    See Also
    --------
    :func:`AegeanTools.angle_tools.gcd`
    """

    if np.all(src1 == src2):
        return 0
    return gcd(src1.ra, src1.dec, src2.ra, src2.dec)