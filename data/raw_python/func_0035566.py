def validate_config(conf_dict):
    """Validate configuration.

    :param conf_dict: test configuration.
    :type conf_dict: {}
    :raise InvalidConfigurationError:
    """
    # TASK improve validation
    if APPLICATIONS not in conf_dict.keys():
        raise InvalidConfigurationError('Missing application configuration.')
    if SEED_FILES not in conf_dict.keys():
        raise InvalidConfigurationError('Missing seed file configuration.')
    if RUNS not in conf_dict.keys():
        conf_dict[RUNS] = DEFAULT_RUNS
    if PROCESSES not in conf_dict.keys():
        conf_dict[PROCESSES] = DEFAULT_PROCESSES
    if PROCESSORS not in conf_dict.keys():
        conf_dict[PROCESSORS] = DEFAULT_PROCESSORS
    return