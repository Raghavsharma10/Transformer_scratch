def _get_config_dict_from_file(config_path=None):
    """Return value if key exists in file.

    Return empty string ("") if key or file does not exist.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # Default (empty) content will be used if config file does not exist.
    config_content = {}

    # If the config file exists we use that content.
    if os.path.isfile(config_path):
        with open(config_path) as fh:
            config_content = json.load(fh)

    return config_content