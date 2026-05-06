def load_config(config_file=None, profile=None, client=None,
                endpoint=None, token=None, solver=None, proxy=None):
    """Load D-Wave Cloud Client configuration based on a configuration file.

    Configuration values can be specified in multiple ways, ranked in the following
    order (with 1 the highest ranked):

    1. Values specified as keyword arguments in :func:`load_config()`. These values replace
       values read from a configuration file, and therefore must be **strings**, including float
       values for timeouts, boolean flags (tested for "truthiness"), and solver feature
       constraints (a dictionary encoded as JSON).
    2. Values specified as environment variables.
    3. Values specified in the configuration file.

    Configuration-file format is described in :mod:`dwave.cloud.config`.

    If the location of the configuration file is not specified, auto-detection
    searches for existing configuration files in the standard directories
    of :func:`get_configfile_paths`.

    If a configuration file explicitly specified, via an argument or
    environment variable, does not exist or is unreadable, loading fails with
    :exc:`~dwave.cloud.exceptions.ConfigFileReadError`. Loading fails
    with :exc:`~dwave.cloud.exceptions.ConfigFileParseError` if the file is
    readable but invalid as a configuration file.

    Similarly, if a profile explicitly specified, via an argument or
    environment variable, is not present in the loaded configuration, loading fails
    with :exc:`ValueError`. Explicit profile selection also fails if the configuration
    file is not explicitly specified, detected on the system, or defined via
    an environment variable.

    Environment variables: ``DWAVE_CONFIG_FILE``, ``DWAVE_PROFILE``, ``DWAVE_API_CLIENT``,
    ``DWAVE_API_ENDPOINT``, ``DWAVE_API_TOKEN``, ``DWAVE_API_SOLVER``, ``DWAVE_API_PROXY``.

    Environment variables are described in :mod:`dwave.cloud.config`.

    Args:

        config_file (str/[str]/None/False/True, default=None):
            Path to configuration file(s).

            If `None`, the value is taken from `DWAVE_CONFIG_FILE` environment
            variable if defined. If the environment variable is undefined or empty,
            auto-detection searches for existing configuration files in the standard
            directories of :func:`get_configfile_paths`.

            If `False`, loading from file(s) is skipped; if `True`, forces auto-detection
            (regardless of the `DWAVE_CONFIG_FILE` environment variable).

        profile (str, default=None):
            Profile name (name of the profile section in the configuration file).

            If undefined, inferred from `DWAVE_PROFILE` environment variable if
            defined. If the environment variable is undefined or empty, a profile is
            selected in the following order:

            1. From the default section if it includes a profile key.
            2. The first section (after the default section).
            3. If no other section is defined besides `[defaults]`, the defaults
               section is promoted and selected.

        client (str, default=None):
            Client type used for accessing the API. Supported values are `qpu`
            for :class:`dwave.cloud.qpu.Client` and `sw` for
            :class:`dwave.cloud.sw.Client`.

        endpoint (str, default=None):
            API endpoint URL.

        token (str, default=None):
            API authorization token.

        solver (str, default=None):
            :term:`solver` features, as a JSON-encoded dictionary of feature constraints,
            the client should use. See :meth:`~dwave.cloud.client.Client.get_solvers` for
            semantics of supported feature constraints.

            If undefined, the client uses a solver definition from environment variables,
            a configuration file, or falls back to the first available online solver.

            For backward compatibility, solver name in string format is accepted and
            converted to ``{"name": <solver name>}``.

        proxy (str, default=None):
            URL for proxy to use in connections to D-Wave API. Can include
            username/password, port, scheme, etc. If undefined, client
            uses the system-level proxy, if defined, or connects directly to the API.

    Returns:
        dict:
            Mapping of configuration keys to values for the profile
            (section), as read from the configuration file and optionally overridden by
            environment values and specified keyword arguments.
            Always contains the `client`, `endpoint`, `token`, `solver`, and `proxy`
            keys.

    Raises:
        :exc:`ValueError`:
            Invalid (non-existing) profile name.

        :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
            Config file parse failed.

    Examples
        This example loads the configuration from an auto-detected configuration file
        in the home directory of a Windows system user.

        >>> import dwave.cloud as dc
        >>> dc.config.load_config()
        {'client': u'qpu',
         'endpoint': u'https://url.of.some.dwavesystem.com/sapi',
         'proxy': None,
         'solver': u'EXAMPLE_2000Q_SYSTEM_A',
         'token': u'DEF-987654321987654321987654321'}
        >>> See which configuration file was loaded
        >>> dc.config.get_configfile_paths()
        [u'C:\\Users\\jane\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf']

        Additional examples are given in :mod:`dwave.cloud.config`.

    """

    if profile is None:
        profile = os.getenv("DWAVE_PROFILE")

    if config_file == False:
        # skip loading from file altogether
        section = {}
    elif config_file == True:
        # force auto-detection, disregarding DWAVE_CONFIG_FILE
        section = load_profile_from_files(None, profile)
    else:
        # auto-detect if not specified with arg or env
        if config_file is None:
            # note: both empty and undefined DWAVE_CONFIG_FILE treated as None
            config_file = os.getenv("DWAVE_CONFIG_FILE")

        # handle ''/None/str/[str] for `config_file` (after env)
        filenames = None
        if config_file:
            if isinstance(config_file, six.string_types):
                filenames = [config_file]
            else:
                filenames = config_file

        section = load_profile_from_files(filenames, profile)

    # override a selected subset of values via env or kwargs,
    # pass-through the rest unmodified
    section['client'] = client or os.getenv("DWAVE_API_CLIENT", section.get('client'))
    section['endpoint'] = endpoint or os.getenv("DWAVE_API_ENDPOINT", section.get('endpoint'))
    section['token'] = token or os.getenv("DWAVE_API_TOKEN", section.get('token'))
    section['solver'] = solver or os.getenv("DWAVE_API_SOLVER", section.get('solver'))
    section['proxy'] = proxy or os.getenv("DWAVE_API_PROXY", section.get('proxy'))

    return section