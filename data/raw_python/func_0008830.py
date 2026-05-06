def save_region(region, filename):
    """
    Save the given region to a file

    Parameters
    ----------
    region : :class:`AegeanTools.regions.Region`
        A region.

    filename : str
        Output file name.
    """
    region.save(filename)
    logging.info("Wrote {0}".format(filename))
    return