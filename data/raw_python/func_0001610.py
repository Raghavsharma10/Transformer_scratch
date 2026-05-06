def load_config(path=None, defaults=None):
    """
    Loads and parses an INI style configuration file using Python's built-in
    configparser module. If path is specified, load it.
    If ``defaults`` (a list of strings) is given, try to load each entry as a
    file, without throwing any error if the operation fails.
    If ``defaults`` is not given, the following locations listed in the
    DEFAULT_FILES constant are tried.
    To completely disable defaults loading, pass in an empty list or ``False``.
    Returns the SafeConfigParser instance used to load and parse the files.
    """

    if defaults is None:
        defaults = DEFAULT_FILES

    config = ConfigParser(allow_no_value=True)

    if defaults:
        config.read(defaults)

    if path:
        with open(path) as fh:
            config.read_file(fh)

    return config