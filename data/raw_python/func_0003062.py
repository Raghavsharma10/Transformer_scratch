def map_crt_to_pol(aryXCrds, aryYrds):
    """Remap coordinates from cartesian to polar

    Parameters
    ----------
    aryXCrds : 1D numpy array
        Array with x coordinate values.
    aryYrds : 1D numpy array
        Array with y coordinate values.

    Returns
    -------
    aryTht : 1D numpy array
        Angle of coordinates
    aryRad : 1D numpy array
        Radius of coordinates.
    """

    aryRad = np.sqrt(aryXCrds**2+aryYrds**2)
    aryTht = np.arctan2(aryYrds, aryXCrds)

    return aryTht, aryRad