def poly2poly(line):
    """
    Parse a string of text containing a DS9 description of a polygon.

    This function works but is not very robust due to the constraints of healpy.

    Parameters
    ----------
    line : str
        A string containing a DS9 region command for a polygon.

    Returns
    -------
    poly : [ra, dec, ...]
        The coordinates of the polygon.
    """
    words = re.split('[(\s,)]', line)
    ras = np.array(words[1::2])
    decs = np.array(words[2::2])
    coords = []
    for ra, dec in zip(ras, decs):
        if ra.strip() == '' or dec.strip() == '':
            continue
        if ":" in ra:
            pos = SkyCoord(Angle(ra, unit=u.hour), Angle(dec, unit=u.degree))
        else:
            pos = SkyCoord(Angle(ra, unit=u.degree), Angle(dec, unit=u.degree))
        # only add this point if it is some distance from the previous one
        coords.extend([pos.ra.degree, pos.dec.degree])
    return coords