def load_profile_from_files(filenames=None, profile=None):
    """Load a profile from a list of D-Wave Cloud Client configuration files.

    .. note:: This method is not standardly used to set up D-Wave Cloud Client configuration.
        It is recommended you use :meth:`.Client.from_config` or
        :meth:`.config.load_config` instead.

    Configuration files comply with standard Windows INI-like format,
    parsable with Python's :mod:`configparser`.

    Each file in the list is progressively searched until the first profile is found.
    This function does not input profile information from environment variables.

    Args:
        filenames (list[str], default=None):
            D-Wave cloud client configuration files (path and name). If ``None``,
            searches for existing configuration files in the standard directories
            of :func:`get_configfile_paths`.

        profile (str, default=None):
            Name of profile to return from reading the configuration from the specified
            configuration file(s). If ``None``, progressively falls back in the
            following order:

            (1) ``profile`` key following ``[defaults]`` section.
            (2) First non-``[defaults]`` section.
            (3) ``[defaults]`` section.

    Returns:
        dict:
            Mapping of configuration keys to values. If no valid config/profile
            is found, returns an empty dict.

    Raises:
        :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
            Config file parse failed.

        :exc:`ValueError`:
            Profile name not found.

    Examples:
        This example loads a profile based on configurations from two files. It
        finds the first profile, dw2000a, in the first file, dwave_a.conf, and adds to
        the values of the defaults section, overwriting the existing client value,
        while ignoring the profile in the second file, dwave_b.conf.

        The files, which are located in the current working directory, are
        (1) dwave_a.conf::

            [defaults]
            endpoint = https://url.of.some.dwavesystem.com/sapi
            client = qpu
            token = ABC-123456789123456789123456789

            [dw2000a]
            client = sw
            solver = EXAMPLE_2000Q_SYSTEM_A
            token = DEF-987654321987654321987654321

        and (2) dwave_b.conf::

            [dw2000b]
            endpoint = https://url.of.some.other.dwavesystem.com/sapi
            client = qpu
            solver = EXAMPLE_2000Q_SYSTEM_B

        The following example code loads profile values from parsing both these files,
        by default loading the first profile encountered or an explicitly specified profile.

        >>> import dwave.cloud as dc
        >>> dc.config.load_profile_from_files(["./dwave_a.conf", "./dwave_b.conf"])   # doctest: +SKIP
        {'client': u'sw',
         'endpoint': u'https://url.of.some.dwavesystem.com/sapi',
         'solver': u'EXAMPLE_2000Q_SYSTEM_A',
         'token': u'DEF-987654321987654321987654321'}
        >>> dc.config.load_profile_from_files(["./dwave_a.conf", "./dwave_b.conf"],
        ...                                   profile='dw2000b')   # doctest: +SKIP
        {'client': u'qpu',
        'endpoint': u'https://url.of.some.other.dwavesystem.com/sapi',
        'solver': u'EXAMPLE_2000Q_SYSTEM_B',
        'token': u'ABC-123456789123456789123456789'}

    """

    # progressively build config from a file, or a list of auto-detected files
    # raises ConfigFileReadError/ConfigFileParseError on error
    config = load_config_from_files(filenames)

    # determine profile name fallback:
    #  (1) profile key under [defaults],
    #  (2) first non-[defaults] section
    #  (3) [defaults] section
    first_section = next(iter(config.sections() + [None]))
    config_defaults = config.defaults()
    if not profile:
        profile = config_defaults.get('profile', first_section)

    if profile:
        try:
            section = dict(config[profile])
        except KeyError:
            raise ValueError("Config profile {!r} not found".format(profile))
    else:
        # as the very last resort (unspecified profile name and
        # no profiles defined in config), try to use [defaults]
        if config_defaults:
            section = config_defaults
        else:
            section = {}

    return section