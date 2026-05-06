def _find_filepath_in_roots(filename):
    """Look for filename in all MEDIA_ROOTS, and return the first one found."""
    for root in settings.DJANGO_STATIC_MEDIA_ROOTS:
        filepath = _filename2filepath(filename, root)
        if os.path.isfile(filepath):
            return filepath, root
    # havent found it in DJANGO_STATIC_MEDIA_ROOTS look for apps' files if we're
    #  in DEBUG mode
    if settings.DEBUG:
        try:
            from django.contrib.staticfiles import finders
            absolute_path = finders.find(filename)
            if absolute_path:
                root, filepath = os.path.split(absolute_path)
                return absolute_path, root
        except ImportError:
            pass
    return None, None