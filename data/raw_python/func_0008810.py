def dec2hms(x):
    """
    Convert decimal degrees into a sexagessimal string in hours.

    Parameters
    ----------
    x : float
        Angle in degrees

    Returns
    -------
    dms : string
        String of format HH:MM:SS.SS
        or XX:XX:XX.XX if x is not finite.
    """
    if not np.isfinite(x):
        return 'XX:XX:XX.XX'
    # wrap negative RA's
    if x < 0:
        x += 360
    x /= 15.0
    h = int(x)
    x = (x - h) * 60
    m = int(x)
    s = (x - m) * 60
    return '{0:02d}:{1:02d}:{2:05.2f}'.format(h, m, s)