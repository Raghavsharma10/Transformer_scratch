def setup_user_config(log):
    """Setup a configuration file in the user's home directory.

    Currently this method stores default values to a fixed configuration
    filename.  It should be modified to run an interactive prompt session
    asking for parameters (or at least confirming the default ones).

    Arguments
    ---------
    log : `logging.Logger` object

    """
    log.warning("AstroCats Setup")
    log.warning("Configure filepath: '{}'".format(_CONFIG_PATH))

    # Create path to configuration file as needed
    config_path_dir = os.path.split(_CONFIG_PATH)[0]
    if not os.path.exists(config_path_dir):
        log.debug("Creating config directory '{}'".format(config_path_dir))
        os.makedirs(config_path_dir)

    if not os.path.isdir(config_path_dir):
        log_raise(log, "Configure path error '{}'".format(config_path_dir))

    # Determine default settings

    # Get this containing directory and use that as default data path
    def_base_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    log.warning("Setting '{}' to default path: '{}'".format(_BASE_PATH_KEY,
                                                            def_base_path))
    config = {_BASE_PATH_KEY: def_base_path}

    # Write settings to configuration file
    json.dump(config, open(_CONFIG_PATH, 'w'))
    if not os.path.exists(def_base_path):
        log_raise(log, "Problem creating configuration file.")

    return