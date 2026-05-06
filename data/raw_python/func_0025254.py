def from_yaml():
    """ Load configuration from yaml source(s), cached to only run once """
    default_yaml_str = snippets.get_snippet_content('hatchery.yml')
    ret = yaml.load(default_yaml_str, Loader=yaml.RoundTripLoader)
    for config_path in CONFIG_LOCATIONS:
        config_path = os.path.expanduser(config_path)
        if os.path.isfile(config_path):
            with open(config_path) as config_file:
                config_dict = yaml.load(config_file, Loader=yaml.RoundTripLoader)
                if config_dict is None:
                    continue
                for k, v in config_dict.items():
                    if k not in ret.keys():
                        raise ConfigError(
                            'found garbage key "{}" in {}'.format(k, config_path)
                        )
                    ret[k] = v
    return ret