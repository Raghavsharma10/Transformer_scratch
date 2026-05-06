def celery_enabled():
    """
    Return a boolean if Celery tasks are enabled for this app.

    If the ``GLITTER_PUBLISHER_CELERY`` setting is ``True`` or ``False`` - then that value will be
    used. However if the setting isn't defined, then this will be enabled automatically if Celery
    is installed.
    """
    enabled = getattr(settings, 'GLITTER_PUBLISHER_CELERY', None)

    if enabled is None:
        try:
            import celery  # noqa
            enabled = True
        except ImportError:
            enabled = False

    return enabled