def get_config_value_from_file(key, config_path=None, default=None):
    """Return value if key exists in file.

    Return default if key not in config.
    """
    config = _get_config_dict_from_file(config_path)
    if key not in config:
        return default
    return config[key]