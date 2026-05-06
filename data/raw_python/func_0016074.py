def init_requests_cache(refresh_cache=False):
    """
    Initializes a cache which the ``requests`` library will consult for
    responses, before making network requests.

    :param refresh_cache: Whether the cache should be cleared out
    """
    # Cache data from external sources; used in some checks
    dirs = AppDirs("stix2-validator", "OASIS")
    # Create cache dir if doesn't exist
    try:
        os.makedirs(dirs.user_cache_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    requests_cache.install_cache(
        cache_name=os.path.join(dirs.user_cache_dir, 'py{}cache'.format(
            sys.version_info[0])),
        expire_after=datetime.timedelta(weeks=1))

    if refresh_cache:
        clear_requests_cache()