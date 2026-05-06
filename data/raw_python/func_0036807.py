def _get_whitelist_licenses(config_path):
    # type: (str) -> List[str]
    """Get whitelist license names from config file.

    :param config_path: str
    :return: list
    """

    whitelist_licenses = []

    try:
        print('config path', config_path)
        with open(config_path) as config:
            whitelist_licenses = [line.rstrip() for line in config]
    except IOError:  # pragma: no cover
        print('Warning: No {} file was found.'.format(LICENSE_CHECKER_CONFIG_NAME))

    return whitelist_licenses