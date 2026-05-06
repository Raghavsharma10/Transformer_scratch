def get_beam(header):
    """
    Create a :class:`AegeanTools.fits_image.Beam` object from a fits header.

    BPA may be missing but will be assumed to be zero.

    if BMAJ or BMIN are missing then return None instead of a beam object.

    Parameters
    ----------
    header : HDUHeader
        The fits header.

    Returns
    -------
    beam : :class:`AegeanTools.fits_image.Beam`
        Beam object, with a, b, and pa in degrees.
    """

    if "BPA" not in header:
        log.warning("BPA not present in fits header, using 0")
        bpa = 0
    else:
        bpa = header["BPA"]

    if "BMAJ" not in header:
        log.warning("BMAJ not present in fits header.")
        bmaj = None
    else:
        bmaj = header["BMAJ"]

    if "BMIN" not in header:
        log.warning("BMIN not present in fits header.")
        bmin = None
    else:
        bmin = header["BMIN"]
    if None in [bmaj, bmin, bpa]:
        return None
    beam = Beam(bmaj, bmin, bpa)
    return beam