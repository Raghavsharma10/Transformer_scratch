def write_config_value_to_file(key, value, config_path=None):
    """Write key/value pair to config file.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # Get existing config.
    config = _get_config_dict_from_file(config_path)

    # Add/update the key/value pair.
    config[key] = value

    # Create parent directories if they are missing.
    mkdir_parents(os.path.dirname(config_path))

    # Write the content
    with open(config_path, "w") as fh:
        json.dump(config, fh, sort_keys=True, indent=2)

    # Set 600 permissions on the config file.
    os.chmod(config_path, 33216)

    return get_config_value_from_file(key, config_path)