def from_config(cls, config_file=None, profile=None, client=None,
                    endpoint=None, token=None, solver=None, proxy=None,
                    legacy_config_fallback=False, **kwargs):
        """Client factory method to instantiate a client instance from configuration.

        Configuration values can be specified in multiple ways, ranked in the following
        order (with 1 the highest ranked):

        1. Values specified as keyword arguments in :func:`from_config()`
        2. Values specified as environment variables
        3. Values specified in the configuration file

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
                Path to configuration file.

                If ``None``, the value is taken from ``DWAVE_CONFIG_FILE`` environment
                variable if defined. If the environment variable is undefined or empty,
                auto-detection searches for existing configuration files in the standard
                directories of :func:`get_configfile_paths`.

                If ``False``, loading from file is skipped; if ``True``, forces auto-detection
                (regardless of the ``DWAVE_CONFIG_FILE`` environment variable).

            profile (str, default=None):
                Profile name (name of the profile section in the configuration file).

                If undefined, inferred from ``DWAVE_PROFILE`` environment variable if
                defined. If the environment variable is undefined or empty, a profile is
                selected in the following order:

                1. From the default section if it includes a profile key.
                2. The first section (after the default section).
                3. If no other section is defined besides ``[defaults]``, the defaults
                   section is promoted and selected.

            client (str, default=None):
                Client type used for accessing the API. Supported values are ``qpu``
                for :class:`dwave.cloud.qpu.Client` and ``sw`` for
                :class:`dwave.cloud.sw.Client`.

            endpoint (str, default=None):
                API endpoint URL.

            token (str, default=None):
                API authorization token.

            solver (dict/str, default=None):
                Default :term:`solver` features to use in :meth:`~dwave.cloud.client.Client.get_solver`.

                Defined via dictionary of solver feature constraints
                (see :meth:`~dwave.cloud.client.Client.get_solvers`).
                For backward compatibility, a solver name, as a string,
                is also accepted and converted to ``{"name": <solver name>}``.

                If undefined, :meth:`~dwave.cloud.client.Client.get_solver` uses a
                solver definition from environment variables, a configuration file, or
                falls back to the first available online solver.

            proxy (str, default=None):
                URL for proxy to use in connections to D-Wave API. Can include
                username/password, port, scheme, etc. If undefined, client
                uses the system-level proxy, if defined, or connects directly to the API.

            legacy_config_fallback (bool, default=False):
                If True and loading from a standard D-Wave Cloud Client configuration
                file (``dwave.conf``) fails, tries loading a legacy configuration file (``~/.dwrc``).

        Other Parameters:
            Unrecognized keys (str):
                All unrecognized keys are passed through to the appropriate client class constructor
                as string keyword arguments.

                An explicit key value overrides an identical user-defined key value loaded from a
                configuration file.

        Returns:
            :class:`~dwave.cloud.client.Client` (:class:`dwave.cloud.qpu.Client` or :class:`dwave.cloud.sw.Client`, default=:class:`dwave.cloud.qpu.Client`):
                Appropriate instance of a QPU or software client.

        Raises:
            :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
                Config file specified or detected could not be opened or read.

            :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
                Config file parse failed.

        Examples:

            A variety of examples are given in :mod:`dwave.cloud.config`.

            This example initializes :class:`~dwave.cloud.client.Client` from an
            explicitly specified configuration file, "~/jane/my_path_to_config/my_cloud_conf.conf"::

            >>> from dwave.cloud import Client
            >>> client = Client.from_config(config_file='~/jane/my_path_to_config/my_cloud_conf.conf')  # doctest: +SKIP
            >>> # code that uses client
            >>> client.close()

        """

        # try loading configuration from a preferred new config subsystem
        # (`./dwave.conf`, `~/.config/dwave/dwave.conf`, etc)
        config = load_config(
            config_file=config_file, profile=profile, client=client,
            endpoint=endpoint, token=token, solver=solver, proxy=proxy)
        _LOGGER.debug("Config loaded: %r", config)

        # fallback to legacy `.dwrc` if key variables missing
        if legacy_config_fallback:
            warnings.warn("'legacy_config_fallback' is deprecated, please convert "
                          "your legacy .dwrc file to the new config format.", DeprecationWarning)

            if not config.get('token'):
                config = legacy_load_config(
                    profile=profile, client=client,
                    endpoint=endpoint, token=token, solver=solver, proxy=proxy)
                _LOGGER.debug("Legacy config loaded: %r", config)

        # manual override of other (client-custom) arguments
        config.update(kwargs)

        from dwave.cloud import qpu, sw
        _clients = {'qpu': qpu.Client, 'sw': sw.Client, 'base': cls}
        _client = config.pop('client', None) or 'base'

        _LOGGER.debug("Final config used for %s.Client(): %r", _client, config)
        return _clients[_client](**config)