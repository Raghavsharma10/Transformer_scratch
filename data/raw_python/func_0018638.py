def get_dattype_regionmode(regions, scen7=False):
    """
    Get the THISFILE_DATTYPE and THISFILE_REGIONMODE flags for a given region set.

    In all MAGICC input files, there are two flags: THISFILE_DATTYPE and
    THISFILE_REGIONMODE. These tell MAGICC how to read in a given input file. This
    function maps the regions which are in a given file to the value of these flags
    expected by MAGICC.

    Parameters
    ----------
    regions : list_like
        The regions to get THISFILE_DATTYPE and THISFILE_REGIONMODE flags for.

    scen7 : bool, optional
        Whether the file we are getting the flags for is a SCEN7 file or not.

    Returns
    -------
    dict
        Dictionary where the flags are the keys and the values are the value they
        should be set to for the given inputs.
    """
    dattype_flag = "THISFILE_DATTYPE"
    regionmode_flag = "THISFILE_REGIONMODE"
    region_dattype_row = _get_dattype_regionmode_regions_row(regions, scen7=scen7)

    dattype = DATTYPE_REGIONMODE_REGIONS[dattype_flag.lower()][region_dattype_row].iloc[
        0
    ]
    regionmode = DATTYPE_REGIONMODE_REGIONS[regionmode_flag.lower()][
        region_dattype_row
    ].iloc[0]

    return {dattype_flag: dattype, regionmode_flag: regionmode}