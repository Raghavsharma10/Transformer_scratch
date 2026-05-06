def collectstatic(force=False):
    """
    collect static files for production httpd

    If run with ``settings.DEBUG==True``, this is a no-op
    unless ``force`` is set to ``True``
    """
    # noise reduction: only collectstatic if not in debug mode
    from django.conf import settings
    if force or not settings.DEBUG:
        tasks.manage('collectstatic', '--noinput')
        print('... finished collectstatic')
        print('')
    else:
        print('... skipping collectstatic as settings.DEBUG=True; If you want to generate staticfiles anyway, run ape collectstatic instead;')