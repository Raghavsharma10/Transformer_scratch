def load_configuration(conf_path):
    """Load and validate test configuration.

    :param conf_path: path to YAML configuration file.
    :return: configuration as dict.
    """
    with open(conf_path) as f:
        conf_dict = yaml.load(f)
    validate_config(conf_dict)
    return conf_dict