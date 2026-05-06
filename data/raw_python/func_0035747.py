def get_config_value(key, config_path=None, default=None):
    """Get a configuration value.

    Preference:
    1. From environment
    2. From JSON configuration file supplied in ``config_path`` argument
    3. The default supplied to the function

    :param key: name of lookup value
    :param config_path: path to JSON configuration file
    :param default: default fall back value
    :returns: value associated with the key
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # Start by setting default value
    value = default

    # Update from config file
    value = get_config_value_from_file(
        key=key,
        config_path=config_path,
        default=value
    )

    # Update from environment variable
    value = os.environ.get(key, value)
    return value