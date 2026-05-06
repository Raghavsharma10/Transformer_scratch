def _check_backend():
    """
    Check :py:class:`djconfig.middleware.DjConfigMiddleware`\
    is registered into ``settings.MIDDLEWARE_CLASSES``
    """
    # Django 1.10 does not allow
    # both settings to be set
    middleware = set(
        getattr(settings, 'MIDDLEWARE', None) or
        getattr(settings, 'MIDDLEWARE_CLASSES', None) or
        [])

    # Deprecated alias
    if "djconfig.middleware.DjConfigLocMemMiddleware" in middleware:
        return

    if "djconfig.middleware.DjConfigMiddleware" in middleware:
        return

    raise ValueError(
        "djconfig.middleware.DjConfigMiddleware "
        "is required but it was not found in "
        "MIDDLEWARE_CLASSES nor in MIDDLEWARE")