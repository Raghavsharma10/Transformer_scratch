def get_pylint_options(config_dir='.'):
    # type: (str) -> List[str]
    """Checks for local config overrides for `pylint`
    and add them in the correct `pylint` `options` format.

    :param config_dir:
    :return: List [str]
    """
    if PYLINT_CONFIG_NAME in os.listdir(config_dir):
        pylint_config_path = PYLINT_CONFIG_NAME
    else:
        pylint_config_path = DEFAULT_PYLINT_CONFIG_PATH

    return ['--rcfile={}'.format(pylint_config_path)]