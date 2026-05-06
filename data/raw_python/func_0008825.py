def circle2circle(line):
    """
    Parse a string that describes a circle in ds9 format.

    Parameters
    ----------
    line : str
        A string containing a DS9 region command for a circle.

    Returns
    -------
    circle : [ra, dec, radius]
        The center and radius of the circle.
    """
    words = re.split('[(,\s)]', line)
    ra = words[1]
    dec = words[2]
    radius = words[3][:-1]  # strip the "
    if ":" in ra:
        ra = Angle(ra, unit=u.hour)
    else:
        ra = Angle(ra, unit=u.degree)
    dec = Angle(dec, unit=u.degree)
    radius = Angle(radius, unit=u.arcsecond)
    return [ra.degree, dec.degree, radius.degree]