def inMicrolensRegion(ra_deg, dec_deg, padding=0):
    """Returns `True` if the given sky oordinate falls on the K2C9 superstamp.

    Parameters
    ----------
    ra_deg : float
        Right Ascension (J2000) in decimal degrees.

    dec_deg : float
        Declination (J2000) in decimal degrees.

    padding : float
        Target must be at least `padding` pixels away from the edge of the
        superstamp. (Note that CCD boundaries are not considered as edges
        in this case.)

    Returns
    -------
    onMicrolensRegion : bool
        `True` if the given coordinate is within the K2C9 microlens superstamp.
    """
    fov = getKeplerFov(9)
    try:
        ch, col, row = fov.getChannelColRow(ra_deg, dec_deg,
                                            allowIllegalReturnValues=False)
        return maskInMicrolensRegion(ch, col, row, padding=padding)
    except ValueError:
        return False