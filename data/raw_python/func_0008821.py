def mask_catalog(regionfile, infile, outfile, negate=False, racol='ra', deccol='dec'):
    """
    Apply a region file as a mask to a catalog, removing all the rows with ra/dec inside the region
    If negate=False then remove the rows with ra/dec outside the region.


    Parameters
    ----------
    regionfile : str
        A file which can be loaded as a :class:`AegeanTools.regions.Region`.
        The catalogue will be masked according to this region.

    infile : str
        Input catalogue.

    outfile : str
        Output catalogue.

    negate :  bool
        If True then pixels *outside* the region are masked.
        Default = False.

    racol, deccol : str
        The name of the columns in `table` that should be interpreted as ra and dec.
        Default = 'ra', 'dec'

    See Also
    --------
    :func:`AegeanTools.MIMAS.mask_table`

    :func:`AegeanTools.catalogs.load_table`
    """
    logging.info("Loading region from {0}".format(regionfile))
    region = Region.load(regionfile)
    logging.info("Loading catalog from {0}".format(infile))
    table = load_table(infile)
    masked_table = mask_table(region, table, negate=negate, racol=racol, deccol=deccol)
    write_table(masked_table, outfile)
    return