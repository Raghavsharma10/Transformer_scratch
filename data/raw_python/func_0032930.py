def _patch_static_handler(handler):
    """Patch in support for static files serving if supported and enabled.
    """

    if django.VERSION[:2] < (1, 3):
        return

    from django.contrib.staticfiles.handlers import StaticFilesHandler
    return StaticFilesHandler(handler)