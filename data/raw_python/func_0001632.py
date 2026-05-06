def includeme(config):
    """ Add pyramid_webpack methods and config to the app """
    settings = config.registry.settings
    root_package_name = config.root_package.__name__
    config.registry.webpack = {
        'DEFAULT': WebpackState(settings, root_package_name)
    }
    for extra_config in aslist(settings.get('webpack.configs', [])):
        state = WebpackState(settings, root_package_name, name=extra_config)
        config.registry.webpack[extra_config] = state

    # Set up any static views
    for state in six.itervalues(config.registry.webpack):
        if state.static_view:
            config.add_static_view(name=state.static_view_name,
                                   path=state.static_view_path,
                                   cache_max_age=state.cache_max_age)

    config.add_request_method(get_webpack, 'webpack')