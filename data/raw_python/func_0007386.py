def get_cache_buster(src_path, method='importtime'):
    """ Return a string that can be used as a parameter for cache-busting URLs
    for this asset.

    :param src_path:
        Filesystem path to the file we're generating a cache-busting value for.

    :param method:
        Method for cache-busting. Supported values: importtime, mtime, md5
        The default is 'importtime', because it requires the least processing.

    Note that the mtime and md5 cache busting methods' results are cached on
    the src_path.

    Example::

        >>> SRC_PATH = os.path.join(os.path.dirname(__file__), 'html.py')
        >>> get_cache_buster(SRC_PATH) is _IMPORT_TIME
        True
        >>> get_cache_buster(SRC_PATH, method='mtime') == _cache_key_by_mtime(SRC_PATH)
        True
        >>> get_cache_buster(SRC_PATH, method='md5') == _cache_key_by_md5(SRC_PATH)
        True
    """
    try:
        fn = _BUST_METHODS[method]
    except KeyError:
        raise KeyError('Unsupported busting method value: %s' % method)

    return fn(src_path)