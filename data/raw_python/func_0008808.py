def dec2dec(dec):
    """
    Convert sexegessimal RA string into a float in degrees.

    Parameters
    ----------
    dec : string
        A string separated representing the Dec.
        Expected format is `[+- ]hh:mm[:ss.s]`
        Colons can be replaced with any whit space character.

    Returns
    -------
    dec : float
        The Dec in degrees.
    """
    d = dec.replace(':', ' ').split()
    if len(d) == 2:
        d.append(0.0)
    if d[0].startswith('-') or float(d[0]) < 0:
        return float(d[0]) - float(d[1]) / 60.0 - float(d[2]) / 3600.0
    return float(d[0]) + float(d[1]) / 60.0 + float(d[2]) / 3600.0