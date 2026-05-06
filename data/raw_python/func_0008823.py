def mim2fits(mimfile, fitsfile):
    """
    Convert a MIMAS region (.mim) file into a MOC region (.fits) file.

    Parameters
    ----------
    mimfile : str
        Input file in MIMAS format.

    fitsfile : str
        Output file.
    """
    region = Region.load(mimfile)
    region.write_fits(fitsfile, moctool='MIMAS {0}-{1}'.format(__version__, __date__))
    logging.info("Converted {0} -> {1}".format(mimfile, fitsfile))
    return