def get_magicc6_to_magicc7_variable_mapping(inverse=False):
    """Get the mappings from MAGICC6 to MAGICC7 variables.

    Note that this mapping is not one to one. For example, "HFC4310", "HFC43-10" and
    "HFC-43-10" in MAGICC6 both map to "HFC4310" in MAGICC7 but "HFC4310" in
    MAGICC7 maps back to "HFC4310".

    Note that HFC-245fa was mistakenly labelled as HFC-245ca in MAGICC6. In reality,
    they are not the same thing. However, the MAGICC6 labelling was merely a typo so
    the mapping between the two is one-to-one.

    Parameters
    ----------
    inverse : bool
        If True, return the inverse mappings i.e. MAGICC7 to MAGICC6 mappings

    Returns
    -------
    dict
        Dictionary of mappings
    """
    # we generate the mapping dynamically, the first name in the list
    # is the one which will be used for inverse mappings
    magicc6_simple_mapping_vars = [
        "KYOTO-CO2EQ",
        "CO2I",
        "CO2B",
        "CH4",
        "N2O",
        "BC",
        "OC",
        "SOx",
        "NOx",
        "NMVOC",
        "CO",
        "SF6",
        "NH3",
        "CF4",
        "C2F6",
        "HFC4310",
        "HFC43-10",
        "HFC-43-10",
        "HFC4310",
        "HFC134a",
        "HFC143a",
        "HFC227ea",
        "CCl4",
        "CH3CCl3",
        "HFC245fa",
        "Halon 1211",
        "Halon 1202",
        "Halon 1301",
        "Halon 2402",
        "Halon1211",
        "Halon1202",
        "Halon1301",
        "Halon2402",
        "CH3Br",
        "CH3Cl",
        "C6F14",
    ]

    magicc6_sometimes_hyphen_vars = [
        "CFC-11",
        "CFC-12",
        "CFC-113",
        "CFC-114",
        "CFC-115",
        "HCFC-22",
        "HFC-23",
        "HFC-32",
        "HFC-125",
        "HFC-134a",
        "HFC-143a",
        "HCFC-141b",
        "HCFC-142b",
        "HFC-227ea",
        "HFC-245fa",
    ]
    magicc6_sometimes_hyphen_vars = [
        v.replace("-", "") for v in magicc6_sometimes_hyphen_vars
    ] + magicc6_sometimes_hyphen_vars

    magicc6_sometimes_underscore_vars = [
        "HFC43_10",
        "CFC_11",
        "CFC_12",
        "CFC_113",
        "CFC_114",
        "CFC_115",
        "HCFC_22",
        "HCFC_141b",
        "HCFC_142b",
    ]
    magicc6_sometimes_underscore_replacements = {
        v: v.replace("_", "") for v in magicc6_sometimes_underscore_vars
    }

    special_case_replacements = {
        "FossilCO2": "CO2I",
        "OtherCO2": "CO2B",
        "MCF": "CH3CCL3",
        "CARB_TET": "CCL4",
        "MHALOSUMCFC12EQ": "MHALOSUMCFC12EQ",  # special case to avoid confusion with MCF
    }

    one_way_replacements = {"HFC-245ca": "HFC245FA", "HFC245ca": "HFC245FA"}

    all_possible_magicc6_vars = (
        magicc6_simple_mapping_vars
        + magicc6_sometimes_hyphen_vars
        + magicc6_sometimes_underscore_vars
        + list(special_case_replacements.keys())
        + list(one_way_replacements.keys())
    )
    replacements = {}
    for m6v in all_possible_magicc6_vars:
        if m6v in special_case_replacements:
            replacements[m6v] = special_case_replacements[m6v]
        elif (
            m6v in magicc6_sometimes_underscore_vars and not inverse
        ):  # underscores one way
            replacements[m6v] = magicc6_sometimes_underscore_replacements[m6v]
        elif (m6v in one_way_replacements) and not inverse:
            replacements[m6v] = one_way_replacements[m6v]
        else:
            m7v = m6v.replace("-", "").replace(" ", "").upper()
            # i.e. if we've already got a value for the inverse, we don't
            # want to overwrite it
            if (m7v in replacements.values()) and inverse:
                continue
            replacements[m6v] = m7v

    if inverse:
        return {v: k for k, v in replacements.items()}
    else:
        return replacements