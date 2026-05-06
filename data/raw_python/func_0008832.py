def get_pixinfo(header):
    """
    Return some pixel information based on the given hdu header
    pixarea - the area of a single pixel in deg2
    pixscale - the side lengths of a pixel (assuming they are square)

    Parameters
    ----------
    header : HDUHeader or dict
        FITS header information

    Returns
    -------
    pixarea : float
        The are of a single pixel at the reference location, in square degrees.

    pixscale : (float, float)
        The pixel scale in degrees, at the reference location.

    Notes
    -----
    The reference location is not always at the image center, and the pixel scale/area may
    change over the image, depending on the projection.
    """
    if all(a in header for a in ["CDELT1", "CDELT2"]):
        pixarea = abs(header["CDELT1"]*header["CDELT2"])
        pixscale = (header["CDELT1"], header["CDELT2"])
    elif all(a in header for a in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]):
        pixarea = abs(header["CD1_1"]*header["CD2_2"]
                    - header["CD1_2"]*header["CD2_1"])
        pixscale = (header["CD1_1"], header["CD2_2"])
        if not (header["CD1_2"] == 0 and header["CD2_1"] == 0):
            log.warning("Pixels don't appear to be square -> pixscale is wrong")
    elif all(a in header for a in ["CD1_1", "CD2_2"]):
        pixarea = abs(header["CD1_1"]*header["CD2_2"])
        pixscale = (header["CD1_1"], header["CD2_2"])
    else:
        log.critical("cannot determine pixel area, using zero EVEN THOUGH THIS IS WRONG!")
        pixarea = 0
        pixscale = (0, 0)
    return pixarea, pixscale