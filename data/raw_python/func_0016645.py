def legacy_load_config(profile=None, endpoint=None, token=None, solver=None,
                       proxy=None, **kwargs):
    """Load configured URLs and token for the SAPI server.

    .. warning:: Included only for backward compatibility. Please use
        :func:`load_config` or the client factory
        :meth:`~dwave.cloud.client.Client.from_config` instead.

    This method tries to load a legacy configuration file from ``~/.dwrc``, select a
    specified `profile` (or, if not specified, the first profile), and override
    individual keys with values read from environment variables or
    specified explicitly as key values in the function.

    Configuration values can be specified in multiple ways, ranked in the following
    order (with 1 the highest ranked):

    1. Values specified as keyword arguments in :func:`legacy_load_config()`
    2. Values specified as environment variables
    3. Values specified in the legacy ``~/.dwrc`` configuration file

    Environment variables searched for are:

     - ``DW_INTERNAL__HTTPLINK``
     - ``DW_INTERNAL__TOKEN``
     - ``DW_INTERNAL__HTTPPROXY``
     - ``DW_INTERNAL__SOLVER``

    Legacy configuration file format is a modified CSV where the first comma is
    replaced with a bar character (``|``). Each line encodes a single profile. Its
    columns are::

        profile_name|endpoint_url,authentication_token,proxy_url,default_solver_name

    All its fields after ``authentication_token`` are optional.

    When there are multiple connections in a file, the first one is
    the default. Any commas in the URLs must be percent-encoded.

    Args:
        profile (str):
            Profile name in the legacy configuration file.

        endpoint (str, default=None):
            API endpoint URL.

        token (str, default=None):
            API authorization token.

        solver (str, default=None):
            Default solver to use in :meth:`~dwave.cloud.client.Client.get_solver`.
            If undefined, all calls to :meth:`~dwave.cloud.client.Client.get_solver`
            must explicitly specify the solver name/id.

        proxy (str, default=None):
            URL for proxy to use in connections to D-Wave API. Can include
            username/password, port, scheme, etc. If undefined, client uses a
            system-level proxy, if defined, or connects directly to the API.

    Returns:
        Dictionary with keys: endpoint, token, solver, and proxy.

    Examples:
        This example creates a client using the :meth:`~dwave.cloud.client.Client.from_config`
        method, which falls back on the legacy file by default when it fails to
        find a D-Wave Cloud Client configuration file (setting its `legacy_config_fallback`
        parameter to False precludes this fall-back operation). For this example,
        no D-Wave Cloud Client configuration file is present on the local system;
        instead the following ``.dwrc`` legacy configuration file is present in the
        user's home directory::

            profile-a|https://one.com,token-one
            profile-b|https://two.com,token-two

        The following example code creates a client without explicitly specifying
        key values, therefore auto-detection searches for existing (non-legacy) configuration
        files in the standard directories of :func:`get_configfile_paths` and, failing to
        find one, falls back on the existing legacy configuration file above.

        >>> import dwave.cloud as dc
        >>> client = dwave.cloud.Client.from_config()    # doctest: +SKIP
        >>> client.endpoint   # doctest: +SKIP
        'https://one.com'
        >>> client.token    # doctest: +SKIP
        'token-one'

        The following examples specify a profile and/or token.

        >>> # Explicitly specify a profile
        >>> client = dwave.cloud.Client.from_config(profile='profile-b')
        >>> # Will try to connect with the url `https://two.com` and the token `token-two`.
        >>> client = dwave.cloud.Client.from_config(profile='profile-b', token='new-token')
        >>> # Will try to connect with the url `https://two.com` and the token `new-token`.

    """

    def _parse_config(fp, filename):
        fields = ('endpoint', 'token', 'proxy', 'solver')
        config = OrderedDict()
        for line in fp:
            # strip whitespace, skip blank and comment lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # parse each record, store in dict with label as key
            try:
                label, data = line.split('|', 1)
                values = [v.strip() or None for v in data.split(',')]
                config[label] = dict(zip(fields, values))
            except:
                raise ConfigFileParseError(
                    "Failed to parse {!r}, line {!r}".format(filename, line))
        return config

    def _read_config(filename):
        try:
            with open(filename, 'r') as f:
                return _parse_config(f, filename)
        except (IOError, OSError):
            raise ConfigFileReadError("Failed to read {!r}".format(filename))

    config = {}
    filename = os.path.expanduser('~/.dwrc')
    if os.path.exists(filename):
        config = _read_config(filename)

    # load profile if specified, or first one in file
    if profile:
        try:
            section = config[profile]
        except KeyError:
            raise ValueError("Config profile {!r} not found".format(profile))
    else:
        try:
            _, section = next(iter(config.items()))
        except StopIteration:
            section = {}

    # override config variables (if any) with environment and then with arguments
    section['endpoint'] = endpoint or os.getenv("DW_INTERNAL__HTTPLINK", section.get('endpoint'))
    section['token'] = token or os.getenv("DW_INTERNAL__TOKEN", section.get('token'))
    section['proxy'] = proxy or os.getenv("DW_INTERNAL__HTTPPROXY", section.get('proxy'))
    section['solver'] = solver or os.getenv("DW_INTERNAL__SOLVER", section.get('solver'))
    section.update(kwargs)

    return section