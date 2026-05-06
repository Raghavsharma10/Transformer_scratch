def configure_uwsgi(configurator_func):
    """Allows configuring uWSGI using Configuration objects returned
    by the given configuration function.

    .. code-block: python

        # In configuration module, e.g `uwsgicfg.py`

        from uwsgiconf.config import configure_uwsgi

        configure_uwsgi(get_configurations)


    :param callable configurator_func: Function which return a list on configurations.

    :rtype: list|None

    :returns: A list with detected configurations or
        ``None`` if called from within uWSGI (e.g. when trying to load WSGI application).

    :raises ConfigurationError:

    """
    from .settings import ENV_CONF_READY, ENV_CONF_ALIAS, CONFIGS_MODULE_ATTR

    if os.environ.get(ENV_CONF_READY):
        # This call is from uWSGI trying to load an application.

        # We prevent unnecessary configuration
        # for setups where application is located in the same
        # file as configuration.

        del os.environ[ENV_CONF_READY]  # Drop it support consecutive reconfiguration.

        return None

    configurations = configurator_func()
    registry = OrderedDict()

    if not isinstance(configurations, (list, tuple)):
        configurations = [configurations]

    for conf_candidate in configurations:
        if not isinstance(conf_candidate, (Section, Configuration)):
            continue

        if isinstance(conf_candidate, Section):
            conf_candidate = conf_candidate.as_configuration()

        alias = conf_candidate.alias

        if alias in registry:
            raise ConfigurationError(
                "Configuration alias '%s' clashes with another configuration. "
                "Please change the alias." % alias)

        registry[alias] = conf_candidate

    if not registry:
        raise ConfigurationError(
            "Callable passed into 'configure_uwsgi' must return 'Section' or 'Configuration' objects.")

    # Try to get configuration alias from env with fall back
    # to --conf argument (as passed by UwsgiRunner.spawn()).
    target_alias = os.environ.get(ENV_CONF_ALIAS)

    if not target_alias:
        last = sys.argv[-2:]
        if len(last) == 2 and last[0] == '--conf':
            target_alias = last[1]

    conf_list = list(registry.values())

    if target_alias:
        # This call is [presumably] from uWSGI configuration read procedure.
        config = registry.get(target_alias)

        if config:
            section = config.sections[0]  # type: Section
            # Set ready marker which is checked above.
            os.environ[ENV_CONF_READY] = '1'

            # Placeholder for runtime introspection.
            section.set_placeholder('config-alias', target_alias)

            # Print out
            config.print_ini()

    else:
        # This call is from module containing uWSGI configurations.
        import inspect

        # Set module attribute automatically.
        config_module = inspect.currentframe().f_back
        config_module.f_locals[CONFIGS_MODULE_ATTR] = conf_list

    return conf_list