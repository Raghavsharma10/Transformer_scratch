def read_config(config_path_or_dict=None):
    """
    Read config from given path string or dict object.

    :param config_path_or_dict:
    :type config_path_or_dict: str or dict
    :return: Returns config object or None if not found.
    :rtype: :class:`revision.config.Config`
    """
    config = None

    if isinstance(config_path_or_dict, dict):
        config = Config(config_path_or_dict)

    if isinstance(config_path_or_dict, string_types):
        if os.path.isabs(config_path_or_dict):
            config_path = config_path_or_dict
        else:
            config_path = os.path.join(
                os.getcwd(),
                os.path.normpath(config_path_or_dict)
            )
    else:
        config_path = os.path.join(
            os.getcwd(),
            DEFAULT_CONFIG_PATH
        )

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
            config = Config(data)

    if config is None:
        raise ConfigNotFound()
    else:
        config.validate()

        return config