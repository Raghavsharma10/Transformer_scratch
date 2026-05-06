def get_special_scen_code(regions, emissions):
    """
    Get special code for MAGICC6 SCEN files.

    At the top of every MAGICC6 and MAGICC5 SCEN file there is a two digit
    number. The first digit, the 'scenfile_region_code' tells MAGICC how many regions
    data is being provided for. The second digit, the 'scenfile_emissions_code', tells
    MAGICC which gases are in the SCEN file.

    The variables which are part of ``PART_OF_SCENFILE_WITH_EMISSIONS_CODE_1`` are the
    emissions species which are expected when scenfile_emissions_code is 1. Similarly,
    ``PART_OF_SCENFILE_WITH_EMISSIONS_CODE_0`` defines the emissions species which are
    expected when scenfile_emissions_code is 0.

    Having these definitions allows Pymagicc to check that the right
    set of emissions has been provided before writing SCEN files.

    Parameters
    ----------
    region : list_like
        Regions to get code for.
    emissions : list-like
        Emissions to get code for.

    Raises
    ------
    ValueError
        If the special scen code cannot be determined.

    Returns
    -------
    int
        The special scen code for the regions-emissions combination provided.
    """
    if sorted(set(PART_OF_SCENFILE_WITH_EMISSIONS_CODE_0)) == sorted(set(emissions)):
        scenfile_emissions_code = 0
    elif sorted(set(PART_OF_SCENFILE_WITH_EMISSIONS_CODE_1)) == sorted(set(emissions)):
        scenfile_emissions_code = 1
    else:
        msg = "Could not determine scen special code for emissions {}".format(emissions)
        raise ValueError(msg)

    if set(regions) == set(["WORLD"]):
        scenfile_region_code = 1
    elif set(regions) == set(["WORLD", "OECD90", "REF", "ASIA", "ALM"]):
        scenfile_region_code = 2
    elif set(regions) == set(["WORLD", "R5OECD", "R5REF", "R5ASIA", "R5MAF", "R5LAM"]):
        scenfile_region_code = 3
    elif set(regions) == set(
        ["WORLD", "R5OECD", "R5REF", "R5ASIA", "R5MAF", "R5LAM", "BUNKERS"]
    ):
        scenfile_region_code = 4
    try:
        return scenfile_region_code * 10 + scenfile_emissions_code
    except NameError:
        msg = "Could not determine scen special code for regions {}".format(regions)
        raise ValueError(msg)