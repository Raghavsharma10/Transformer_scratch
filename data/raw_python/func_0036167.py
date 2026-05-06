def build_config(config_file=get_system_config_directory()):
    """
    Construct the config object from necessary elements.
    """
    config = Config(config_file, allow_no_value=True)
    application_versions = find_applications_on_system()

    # Add found versions to config if they don't exist. Versions found
    # in the config file takes precedence over versions found in PATH.
    for item in application_versions.iteritems():
        if not config.has_option(Config.EXECUTABLES, item[0]):
            config.set(Config.EXECUTABLES, item[0], item[1])
    return config