def catalog_to_moc(catalog, radius, order, **kwargs):
    """
    Convert a catalog to a MOC.

    The catalog is given as an Astropy SkyCoord object containing
    multiple coordinates.  The radius of catalog entries can be
    given as an Astropy Quantity (with units), otherwise it is assumed
    to be in arcseconds.

    Any additional keyword arguments are passed on to `catalog_to_cells`.
    """

    # Generate list of MOC cells.
    cells = catalog_to_cells(catalog, radius, order, **kwargs)

    # Create new MOC object using our collection of cells.
    moc = MOC(moctype='CATALOG')
    moc.add(order, cells, no_validation=True)
    return moc