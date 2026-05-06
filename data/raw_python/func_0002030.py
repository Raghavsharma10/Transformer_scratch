def get_config():
    """Read the configfile and return config dict.

    Returns
    -------
    dict
        Dictionary with the content of the configpath file.
    """
    configpath = get_configpath()
    if not configpath.exists():
        raise IOError("Config file {} not found.".format(str(configpath)))
    else:
        config = configparser.ConfigParser()
        config.read(str(configpath))
        return config