def create_config_loader(config=None, env_prefix='APP'):
    """Create a default configuration loader.

    A configuration loader takes a Flask application and keyword arguments and
    updates the Flask application's configuration as it sees fit.

    This default configuration loader will load configuration in the following
    order:

        1. Load configuration from ``invenio_config.module`` entry points
           group, following the alphabetical ascending order in case of
           multiple entry points defined.
           For example, the config of an app with entry point name ``10_app``
           will be loaded after the config of an app with entry point name
           ``00_app``.
        2. Load configuration from ``config`` module if provided as argument.
        3. Load configuration from the instance folder:
           ``<app.instance_path>/<app.name>.cfg``.
        4. Load configuration keyword arguments provided.
        5. Load configuration from environment variables with the prefix
           ``env_prefix``.

    If no secret key has been set a warning will be issued.

    :param config: Either an import string to a module with configuration or
        alternatively the module itself.
    :param env_prefix: Environment variable prefix to import configuration
        from.
    :return: A callable with the method signature
        ``config_loader(app, **kwargs)``.

    .. versionadded:: 1.0.0
    """
    def _config_loader(app, **kwargs_config):
        InvenioConfigEntryPointModule(app=app)
        if config:
            InvenioConfigModule(app=app, module=config)
        InvenioConfigInstanceFolder(app=app)
        app.config.update(**kwargs_config)
        InvenioConfigEnvironment(app=app, prefix='{0}_'.format(env_prefix))
        InvenioConfigDefault(app=app)

    return _config_loader