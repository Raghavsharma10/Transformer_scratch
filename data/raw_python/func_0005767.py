def _load_config(initial_namespace=None, defaults=None):
    # type: (Optional[str], Optional[str]) -> ConfigLoader
    """
    Kwargs:
        initial_namespace:
        defaults:
    """
    # load defaults
    if defaults:
        config = ConfigLoader()
        config.update_from_object(defaults)

    namespace = getattr(config, 'CONFIG_NAMESPACE', initial_namespace)
    app_config = getattr(config, 'APP_CONFIG', None)

    # load customised config
    if app_config:
        if namespace is None:
            config.update_from_object(app_config)
        else:
            _temp = ConfigLoader()
            _temp.update_from_object(app_config, lambda key: key.startswith(namespace))
            config.update(_temp.namespace(namespace))

    return config