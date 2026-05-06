def get_magicc7_to_openscm_variable_mapping(inverse=False):
    """Get the mappings from MAGICC7 to OpenSCM variables.

    Parameters
    ----------
    inverse : bool
        If True, return the inverse mappings i.e. OpenSCM to MAGICC7 mappings

    Returns
    -------
    dict
        Dictionary of mappings
    """

    def get_openscm_replacement(in_var):
        if in_var.endswith("_INVERSE_EMIS"):
            prefix = "Inverse Emissions"
        elif in_var.endswith("_EMIS"):
            prefix = "Emissions"
        elif in_var.endswith("_CONC"):
            prefix = "Atmospheric Concentrations"
        elif in_var.endswith("_RF"):
            prefix = "Radiative Forcing"
        elif in_var.endswith("_OT"):
            prefix = "Optical Thickness"
        else:
            raise ValueError("This shouldn't happen")

        variable = in_var.split("_")[0]
        # I hate edge cases
        if variable.endswith("EQ"):
            variable = variable.replace("EQ", " Equivalent")

        if "GHG" in variable:
            variable = variable.replace("GHG", "Greenhouse Gases")

        if "BIOMASSAER" in variable:
            variable = variable.replace("BIOMASSAER", "Aerosols|MAGICC AFOLU")

        if "CO2CH4N2O" in variable:
            variable = variable.replace("CO2CH4N2O", "CO2, CH4 and N2O")

        aggregate_indicators = {
            "KYOTO": "Kyoto Gases",
            "FGASSUM": "F Gases",
            "MHALOSUM": "Montreal Protocol Halogen Gases",
        }
        for agg_indicator, long_name in aggregate_indicators.items():
            if variable.startswith(agg_indicator):
                stripped_var = variable.replace(agg_indicator, "")
                if stripped_var:
                    variable = DATA_HIERARCHY_SEPARATOR.join([stripped_var, long_name])
                else:
                    variable = long_name

        edge_case_B = variable.upper() in ("HCFC141B", "HCFC142B")
        if variable.endswith("I"):
            variable = DATA_HIERARCHY_SEPARATOR.join(
                [variable[:-1], "MAGICC Fossil and Industrial"]
            )
        elif variable.endswith("B") and not edge_case_B:
            variable = DATA_HIERARCHY_SEPARATOR.join([variable[:-1], "MAGICC AFOLU"])

        case_adjustments = {
            "SOX": "SOx",
            "NOX": "NOx",
            "HFC134A": "HFC134a",
            "HFC143A": "HFC143a",
            "HFC152A": "HFC152a",
            "HFC227EA": "HFC227ea",
            "HFC236FA": "HFC236fa",
            "HFC245FA": "HFC245fa",
            "HFC365MFC": "HFC365mfc",
            "HCFC141B": "HCFC141b",
            "HCFC142B": "HCFC142b",
            "CH3CCL3": "CH3CCl3",
            "CCL4": "CCl4",
            "CH3CL": "CH3Cl",
            "CH2CL2": "CH2Cl2",
            "CHCL3": "CHCl3",
            "CH3BR": "CH3Br",
            "HALON1211": "Halon1211",
            "HALON1301": "Halon1301",
            "HALON2402": "Halon2402",
            "HALON1202": "Halon1202",
            "SOLAR": "Solar",
            "VOLCANIC": "Volcanic",
            "EXTRA": "Extra",
        }
        variable = apply_string_substitutions(variable, case_adjustments)

        return DATA_HIERARCHY_SEPARATOR.join([prefix, variable])

    magicc7_suffixes = ["_EMIS", "_CONC", "_RF", "_OT", "_INVERSE_EMIS"]
    magicc7_base_vars = MAGICC7_EMISSIONS_UNITS.magicc_variable.tolist() + [
        "SOLAR",
        "VOLCANIC",
        "CO2EQ",
        "KYOTOCO2EQ",
        "FGASSUMHFC134AEQ",
        "MHALOSUMCFC12EQ",
        "GHG",
        "KYOTOGHG",
        "FGASSUM",
        "MHALOSUM",
        "BIOMASSAER",
        "CO2CH4N2O",
        "EXTRA",
    ]
    magicc7_vars = [
        base_var + suffix
        for base_var in magicc7_base_vars
        for suffix in magicc7_suffixes
    ]

    replacements = {m7v: get_openscm_replacement(m7v) for m7v in magicc7_vars}

    replacements.update(
        {
            "SURFACE_TEMP": "Surface Temperature",
            "TOTAL_INCLVOLCANIC_RF": "Radiative Forcing",
            "VOLCANIC_ANNUAL_RF": "Radiative Forcing|Volcanic",
            "TOTAL_ANTHRO_RF": "Radiative Forcing|Anthropogenic",
            "TOTAER_DIR_RF": "Radiative Forcing|Aerosols|Direct Effect",
            "CLOUD_TOT_RF": "Radiative Forcing|Aerosols|Indirect Effect",
            "MINERALDUST_RF": "Radiative Forcing|Mineral Dust",
            "STRATOZ_RF": "Radiative Forcing|Stratospheric Ozone",
            "TROPOZ_RF": "Radiative Forcing|Tropospheric Ozone",
            "CH4OXSTRATH2O_RF": "Radiative Forcing|CH4 Oxidation Stratospheric H2O",  # what is this
            "LANDUSE_RF": "Radiative Forcing|Land-use Change",
            "BCSNOW_RF": "Radiative Forcing|Black Carbon on Snow",
            "CO2PF_EMIS": "Land to Air Flux|CO2|MAGICC Permafrost",
            # "CH4PF_EMIS": "Land to Air Flux|CH4|MAGICC Permafrost",  # TODO: test and then add when needed
        }
    )

    agg_ocean_heat_top = "Aggregated Ocean Heat Content"
    heat_content_aggreg_depths = {
        "HEATCONTENT_AGGREG_DEPTH{}".format(i): "{}{}Depth {}".format(
            agg_ocean_heat_top, DATA_HIERARCHY_SEPARATOR, i
        )
        for i in range(1, 4)
    }
    replacements.update(heat_content_aggreg_depths)
    replacements.update({"HEATCONTENT_AGGREG_TOTAL": agg_ocean_heat_top})

    ocean_temp_layer = {
        "OCEAN_TEMP_LAYER_{0:03d}".format(i): "Ocean Temperature{}Layer {}".format(
            DATA_HIERARCHY_SEPARATOR, i
        )
        for i in range(1, 999)
    }
    replacements.update(ocean_temp_layer)

    if inverse:
        return {v: k for k, v in replacements.items()}
    else:
        return replacements