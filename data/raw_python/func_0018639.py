def get_region_order(regions, scen7=False):
    """
    Get the region order expected by MAGICC.

    Parameters
    ----------
    regions : list_like
        The regions to get THISFILE_DATTYPE and THISFILE_REGIONMODE flags for.

    scen7 : bool, optional
        Whether the file we are getting the flags for is a SCEN7 file or not.

    Returns
    -------
    list
        Region order expected by MAGICC for the given region set.
    """
    region_dattype_row = _get_dattype_regionmode_regions_row(regions, scen7=scen7)
    region_order = DATTYPE_REGIONMODE_REGIONS["regions"][region_dattype_row].iloc[0]

    return region_order