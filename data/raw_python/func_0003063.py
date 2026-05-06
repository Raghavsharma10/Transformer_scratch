def map_pol_to_crt(aryTht, aryRad):
    """Remap coordinates from polar to cartesian

    Parameters
    ----------
    aryTht : 1D numpy array
        Angle of coordinates
    aryRad : 1D numpy array
        Radius of coordinates.

    Returns
    -------
    aryXCrds : 1D numpy array
        Array with x coordinate values.
    aryYrds : 1D numpy array
        Array with y coordinate values.
    """

    aryXCrds = aryRad * np.cos(aryTht)
    aryYrds = aryRad * np.sin(aryTht)

    return aryXCrds, aryYrds