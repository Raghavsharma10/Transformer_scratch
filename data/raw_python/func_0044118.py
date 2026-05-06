def join_configs(configs):

    """Join all config files into one config."""

    joined_config = {}

    for config in configs:
        joined_config.update(yaml.load(config))

    return joined_config