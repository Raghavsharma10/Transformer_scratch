def configuration_get_default_folder():
    """
    Return the default folder where user-specific data is stored.
    This depends of the system on which Python is running,
    :return: path to the user-specific configuration data folder
    """
    system = platform.system()
    if system == 'Linux':
        # https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
        sys_config_path = Path(os.getenv('XDG_CONFIG_HOME', os.path.expanduser("~/.config")))
    elif system == 'Windows':
        sys_config_path = Path(os.getenv('APPDATA', ''))
    else:
        log.error('Unknown system: "{system}" (using default configuration path)'.format(system=system))
        sys_config_path = Path()
    log.debug('User-specific system configuration folder="{sys_config_path}"'.format(
        sys_config_path=sys_config_path))
    sys_config = sys_config_path / PROJECT_TITLE
    log.debug('User-specific {project} configuration folder="{sys_config}"'.format(
        project=PROJECT_TITLE, sys_config=sys_config))
    return sys_config