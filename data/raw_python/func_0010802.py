def get_permissions_app_name():
    """
    Gets the app after which smartmin permissions should be installed. This can be specified by PERMISSIONS_APP in the
    Django settings or defaults to the last app with models
    """
    global permissions_app_name

    if not permissions_app_name:
        permissions_app_name = getattr(settings, 'PERMISSIONS_APP', None)

        if not permissions_app_name:
            app_names_with_models = [a.name for a in apps.get_app_configs() if a.models_module is not None]
            if app_names_with_models:
                permissions_app_name = app_names_with_models[-1]

    return permissions_app_name