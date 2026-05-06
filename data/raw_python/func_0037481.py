def update_constants(nmrstar2cfg="", nmrstar3cfg="", resonance_classes_cfg="", spectrum_descriptions_cfg=""):
    """Update constant variables.

    :return: None
    :rtype: :py:obj:`None`
    """
    nmrstar_constants = {}
    resonance_classes = {}
    spectrum_descriptions = {}

    this_directory = os.path.dirname(__file__)

    nmrstar2_config_filepath = os.path.join(this_directory, "conf/constants_nmrstar2.json")
    nmrstar3_config_filepath = os.path.join(this_directory, "conf/constants_nmrstar3.json")
    resonance_classes_config_filepath = os.path.join(this_directory, "conf/resonance_classes.json")
    spectrum_descriptions_config_filepath = os.path.join(this_directory, "conf/spectrum_descriptions.json")

    with open(nmrstar2_config_filepath, "r") as nmrstar2config, open(nmrstar3_config_filepath, "r") as nmrstar3config:
        nmrstar_constants["2"] = json.load(nmrstar2config)
        nmrstar_constants["3"] = json.load(nmrstar3config)

    with open(resonance_classes_config_filepath, "r") as config:
        resonance_classes.update(json.load(config))

    with open(spectrum_descriptions_config_filepath, "r") as config:
        spectrum_descriptions.update(json.load(config))

    if nmrstar2cfg:
        with open(nmrstar2cfg, "r") as nmrstar2config:
            nmrstar_constants["2"].update(json.load(nmrstar2config))

    if nmrstar3cfg:
        with open(nmrstar2cfg, "r") as nmrstar3config:
            nmrstar_constants["3"].update(json.load(nmrstar3config))

    if resonance_classes_cfg:
        with open(nmrstar2cfg, "r") as config:
            resonance_classes.update(json.load(config))

    if spectrum_descriptions_cfg:
        with open(spectrum_descriptions_cfg, "r") as config:
            spectrum_descriptions.update(json.load(config))

    NMRSTAR_CONSTANTS.update(nmrstar_constants)
    RESONANCE_CLASSES.update(resonance_classes)
    SPECTRUM_DESCRIPTIONS.update(spectrum_descriptions)