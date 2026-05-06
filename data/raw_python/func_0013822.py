def set_cache(new_path=None):
    """Simple function to change the cache location.

    `new_path` can be an absolute or relative path. If the directory does not
    exist yet, this function will create it. If None it will set the cache to
    the default cache directory.

    If you are going to change the cache directory, this function should be
    called at the top of your script, before you make any calls to the API.
    This is to avoid duplicate files and excess API calls.

    :param new_path: relative or absolute path to the desired new cache
    directory
    :return: str, str
    """

    global CACHE_DIR, API_CACHE, SPRITE_CACHE

    if new_path is None:
        new_path = get_default_cache()

    CACHE_DIR = safe_make_dirs(os.path.abspath(new_path))
    API_CACHE = os.path.join(CACHE_DIR, 'api.cache')
    SPRITE_CACHE = safe_make_dirs(os.path.join(CACHE_DIR, 'sprite'))

    return CACHE_DIR, API_CACHE, SPRITE_CACHE