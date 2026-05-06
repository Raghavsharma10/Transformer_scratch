def get_root_resource(config):
    """Returns the root resource."""
    app_package_name = get_app_package_name(config)
    return config.registry._root_resources.setdefault(
        app_package_name, Resource(config))