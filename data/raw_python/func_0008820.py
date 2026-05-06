def mask_table(region, table, negate=False, racol='ra', deccol='dec'):
    """
    Apply a given mask (region) to the table, removing all the rows with ra/dec inside the region
    If negate=False then remove the rows with ra/dec outside the region.


    Parameters
    ----------
    region : :class:`AegeanTools.regions.Region`
        Region to mask.

    table : Astropy.table.Table
        Table to be masked.

    negate :  bool
        If True then pixels *outside* the region are masked.
        Default = False.

    racol, deccol : str
        The name of the columns in `table` that should be interpreted as ra and dec.
        Default = 'ra', 'dec'

    Returns
    -------
    masked : Astropy.table.Table
        A view of the given table which has been masked.
    """
    inside = region.sky_within(table[racol], table[deccol], degin=True)
    if not negate:
        mask = np.bitwise_not(inside)
    else:
        mask = inside
    return table[mask]