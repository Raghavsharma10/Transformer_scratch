def dec2dms(x):
    """
    Convert decimal degrees into a sexagessimal string in degrees.

    Parameters
    ----------
    x : float
        Angle in degrees

    Returns
    -------
    dms : string
        String of format [+-]DD:MM:SS.SS
        or XX:XX:XX.XX if x is not finite.
    """
    if not np.isfinite(x):
        return 'XX:XX:XX.XX'
    if x < 0:
        sign = '-'
    else:
        sign = '+'
    x = abs(x)
    d = int(math.floor(x))
    m = int(math.floor((x - d) * 60))
    s = float(( (x - d) * 60 - m) * 60)
    return '{0}{1:02d}:{2:02d}:{3:05.2f}'.format(sign, d, m, s)