def box2poly(line):
    """
    Convert a string that describes a box in ds9 format, into a polygon that is given by the corners of the box

    Parameters
    ----------
    line : str
        A string containing a DS9 region command for a box.

    Returns
    -------
    poly : [ra, dec, ...]
        The corners of the box in clockwise order from top left.
    """
    words = re.split('[(\s,)]', line)
    ra = words[1]
    dec = words[2]
    width = words[3]
    height = words[4]
    if ":" in ra:
        ra = Angle(ra, unit=u.hour)
    else:
        ra = Angle(ra, unit=u.degree)
    dec = Angle(dec, unit=u.degree)
    width = Angle(float(width[:-1])/2, unit=u.arcsecond)  # strip the "
    height = Angle(float(height[:-1])/2, unit=u.arcsecond)  # strip the "
    center = SkyCoord(ra, dec)
    tl = center.ra.degree+width.degree, center.dec.degree+height.degree
    tr = center.ra.degree-width.degree, center.dec.degree+height.degree
    bl = center.ra.degree+width.degree, center.dec.degree-height.degree
    br = center.ra.degree-width.degree, center.dec.degree-height.degree
    return np.ravel([tl, tr, br, bl]).tolist()