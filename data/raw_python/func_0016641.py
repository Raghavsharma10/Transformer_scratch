def get_default_configfile_path():
    """Return the default configuration-file path.

    Typically returns a user-local configuration file; e.g:
    ``~/.config/dwave/dwave.conf``.

    Returns:
        str:
            Configuration file path.

    Examples:
        This example displays the default configuration file on an Ubuntu Unix system
        running IPython 2.7.

        >>> import dwave.cloud as dc
        >>> # Display paths
        >>> dc.config.get_configfile_paths(only_existing=False)   # doctest: +SKIP
        ['/etc/xdg/xdg-ubuntu/dwave/dwave.conf',
         '/usr/share/upstart/xdg/dwave/dwave.conf',
         '/etc/xdg/dwave/dwave.conf',
         '/home/mary/.config/dwave/dwave.conf',
         './dwave.conf']
        >>> # Find default configuration path
        >>> dc.config.get_default_configfile_path()   # doctest: +SKIP
        '/home/mary/.config/dwave/dwave.conf'

    """
    base = homebase.user_config_dir(
        app_author=CONF_AUTHOR, app_name=CONF_APP, roaming=False,
        use_virtualenv=False, create=False)
    path = os.path.join(base, CONF_FILENAME)
    return path