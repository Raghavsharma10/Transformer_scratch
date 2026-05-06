def get_magicc_region_to_openscm_region_mapping(inverse=False):
    """Get the mappings from MAGICC to OpenSCM regions.

    This is not a pure inverse of the other way around. For example, we never provide
    "GLOBAL" as a MAGICC return value because it's unnecesarily confusing when we also
    have "World". Fortunately MAGICC doesn't ever read the name "GLOBAL" so this
    shouldn't matter.

    Parameters
    ----------
    inverse : bool
        If True, return the inverse mappings i.e. MAGICC to OpenSCM mappings
    Returns
    -------
    dict
        Dictionary of mappings
    """

    def get_openscm_replacement(in_region):
        world = "World"
        if in_region in ("WORLD", "GLOBAL"):
            return world
        if in_region in ("BUNKERS"):
            return DATA_HIERARCHY_SEPARATOR.join([world, "Bunkers"])
        elif in_region.startswith(("NH", "SH")):
            in_region = in_region.replace("-", "")
            hem = "Northern Hemisphere" if "NH" in in_region else "Southern Hemisphere"
            if in_region in ("NH", "SH"):
                return DATA_HIERARCHY_SEPARATOR.join([world, hem])

            land_ocean = "Land" if "LAND" in in_region else "Ocean"
            return DATA_HIERARCHY_SEPARATOR.join([world, hem, land_ocean])
        else:
            return DATA_HIERARCHY_SEPARATOR.join([world, in_region])

    # we generate the mapping dynamically, the first name in the list
    # is the one which will be used for inverse mappings
    _magicc_regions = [
        "WORLD",
        "GLOBAL",
        "OECD90",
        "ALM",
        "REF",
        "ASIA",
        "R5ASIA",
        "R5OECD",
        "R5REF",
        "R5MAF",
        "R5LAM",
        "R6OECD90",
        "R6REF",
        "R6LAM",
        "R6MAF",
        "R6ASIA",
        "NHOCEAN",
        "SHOCEAN",
        "NHLAND",
        "SHLAND",
        "NH-OCEAN",
        "SH-OCEAN",
        "NH-LAND",
        "SH-LAND",
        "SH",
        "NH",
        "BUNKERS",
    ]

    replacements = {}
    for magicc_region in _magicc_regions:
        openscm_region = get_openscm_replacement(magicc_region)
        # i.e. if we've already got a value for the inverse, we don't want to overwrite
        if (openscm_region in replacements.values()) and inverse:
            continue
        replacements[magicc_region] = openscm_region

    if inverse:
        return {v: k for k, v in replacements.items()}
    else:
        return replacements