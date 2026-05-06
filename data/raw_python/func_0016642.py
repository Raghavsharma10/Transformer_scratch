def load_config_from_files(filenames=None):
    """Load D-Wave Cloud Client configuration from a list of files.

    .. note:: This method is not standardly used to set up D-Wave Cloud Client configuration.
        It is recommended you use :meth:`.Client.from_config` or
        :meth:`.config.load_config` instead.

    Configuration files comply with standard Windows INI-like format,
    parsable with Python's :mod:`configparser`. A section called
    ``defaults`` contains default values inherited by other sections.

    Each filename in the list (each configuration file loaded) progressively upgrades
    the final configuration, on a key by key basis, per each section.

    Args:
        filenames (list[str], default=None):
            D-Wave Cloud Client configuration files (paths and names).

            If ``None``, searches for a configuration file named ``dwave.conf``
            in all system-wide configuration directories, in the user-local
            configuration directory, and in the current working directory,
            following the user/system configuration paths of :func:`get_configfile_paths`.

    Returns:
        :obj:`~configparser.ConfigParser`:
            :class:`dict`-like mapping of configuration sections (profiles) to
            mapping of per-profile keys holding values.

    Raises:
        :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
            Config file parse failed.

    Examples:
        This example loads configurations from two files. One contains a default
        section with key/values that are overwritten by any profile section that
        contains that key/value; for example, profile dw2000b in file dwave_b.conf
        overwrites the default URL and client type, which profile dw2000a inherits
        from the defaults section, while profile dw2000a overwrites the API token that
        profile dw2000b inherits.

        The files, which are located in the current working directory, are
        (1) dwave_a.conf::

            [defaults]
            endpoint = https://url.of.some.dwavesystem.com/sapi
            client = qpu
            token = ABC-123456789123456789123456789

            [dw2000a]
            solver = EXAMPLE_2000Q_SYSTEM
            token = DEF-987654321987654321987654321

        and (2) dwave_b.conf::

            [dw2000b]
            endpoint = https://url.of.some.other.dwavesystem.com/sapi
            client = sw
            solver = EXAMPLE_2000Q_SYSTEM

        The following example code loads configuration from both these files, with
        the defined overrides and inheritance.

        .. code:: python

            >>> import dwave.cloud as dc
            >>> import sys
            >>> configuration = dc.config.load_config_from_files(["./dwave_a.conf", "./dwave_b.conf"])   # doctest: +SKIP
            >>> configuration.write(sys.stdout)    # doctest: +SKIP
            [defaults]
            endpoint = https://url.of.some.dwavesystem.com/sapi
            client = qpu
            token = ABC-123456789123456789123456789

            [dw2000a]
            solver = EXAMPLE_2000Q_SYSTEM
            token = DEF-987654321987654321987654321

            [dw2000b]
            endpoint = https://url.of.some.other.dwavesystem.com/sapi
            client = sw
            solver = EXAMPLE_2000Q_SYSTEM

    """
    if filenames is None:
        filenames = get_configfile_paths()

    config = configparser.ConfigParser(default_section="defaults")
    for filename in filenames:
        try:
            with open(filename, 'r') as f:
                config.read_file(f, filename)
        except (IOError, OSError):
            raise ConfigFileReadError("Failed to read {!r}".format(filename))
        except configparser.Error:
            raise ConfigFileParseError("Failed to parse {!r}".format(filename))
    return config