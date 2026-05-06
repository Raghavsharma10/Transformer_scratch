def get_flake8_options(config_dir='.'):
    # type: (str) -> List[str]
    """Checks for local config overrides for `flake8`
    and add them in the correct `flake8` `options` format.

    :param config_dir:
    :return: List[str]
    """
    if FLAKE8_CONFIG_NAME in os.listdir(config_dir):
        flake8_config_path = FLAKE8_CONFIG_NAME
    else:
        flake8_config_path = DEFAULT_FLAKE8_CONFIG_PATH

    return ['--config={}'.format(flake8_config_path)]