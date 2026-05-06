def enable_cache():
    """Enable requests library cache."""
    try:
        import requests_cache
    except ImportError as err:
        sys.stderr.write('Failed to enable cache: {0}\n'.format(str(err)))
        return
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    requests_cache.install_cache(CACHE_FILE)