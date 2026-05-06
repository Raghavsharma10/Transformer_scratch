def has_app(app_name):
    """
    Determines whether an app is listed in INSTALLED_APPS or the app registry.
    :param app_name: string
    :return: bool
    """
    if DJANGO_VERSION >= (1, 7):
        from django.apps import apps
        return apps.is_installed(app_name)
    else:
        from django.conf import settings

        return app_name in settings.INSTALLED_APPS