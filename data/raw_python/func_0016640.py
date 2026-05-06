def get_configfile_paths(system=True, user=True, local=True, only_existing=True):
    """Return a list of local configuration file paths.

    Search paths for configuration files on the local system
    are based on homebase_ and depend on operating system; for example, for Linux systems
    these might include ``dwave.conf`` in the current working directory (CWD),
    user-local ``.config/dwave/``, and system-wide ``/etc/dwave/``.

    .. _homebase: https://github.com/dwavesystems/homebase

    Args:
        system (boolean, default=True):
            Search for system-wide configuration files.

        user (boolean, default=True):
            Search for user-local configuration files.

        local (boolean, default=True):
            Search for local configuration files (in CWD).

        only_existing (boolean, default=True):
            Return only paths for files that exist on the local system.

    Returns:
        list[str]:
            List of configuration file paths.

    Examples:
        This example displays all paths to configuration files on a Windows system
        running Python 2.7 and then finds the single existing configuration file.

        >>> import dwave.cloud as dc
        >>> # Display paths
        >>> dc.config.get_configfile_paths(only_existing=False)   # doctest: +SKIP
        [u'C:\\ProgramData\\dwavesystem\\dwave\\dwave.conf',
         u'C:\\Users\\jane\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf',
         '.\\dwave.conf']
        >>> # Find existing files
        >>> dc.config.get_configfile_paths()   # doctest: +SKIP
        [u'C:\\Users\\jane\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf']

    """

    candidates = []

    # system-wide has the lowest priority, `/etc/dwave/dwave.conf`
    if system:
        candidates.extend(homebase.site_config_dir_list(
            app_author=CONF_AUTHOR, app_name=CONF_APP,
            use_virtualenv=False, create=False))

    # user-local will override it, `~/.config/dwave/dwave.conf`
    if user:
        candidates.append(homebase.user_config_dir(
            app_author=CONF_AUTHOR, app_name=CONF_APP, roaming=False,
            use_virtualenv=False, create=False))

    # highest priority (overrides all): `./dwave.conf`
    if local:
        candidates.append(".")

    paths = [os.path.join(base, CONF_FILENAME) for base in candidates]
    if only_existing:
        paths = list(filter(os.path.exists, paths))

    return paths