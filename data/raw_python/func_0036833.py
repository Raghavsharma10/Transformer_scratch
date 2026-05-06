def get_license_checker_config_path(config_dir='.'):
    # type: (str) -> List[str]
    """Checks for local config overrides for license checker,
    if not found it returns the package default.

    :param config_dir:
    :return: str
    """
    if LICENSE_CHECKER_CONFIG_NAME in os.listdir(config_dir):
        license_checker_config_path = LICENSE_CHECKER_CONFIG_NAME
    else:
        license_checker_config_path = DEFAULT_LICENSE_CHECKER_CONFIG_PATH

    return license_checker_config_path