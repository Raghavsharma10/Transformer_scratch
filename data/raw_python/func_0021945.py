def get_cache_dir(cache_dir, cache_root_dir=None, *args, **kwargs):
    """
    Return full cache_dir, according to following logic:
        - if cache_dir is a full path (per path.isabs), return that value
        - if not and if cache_root_dir is not None, join two paths
        - otherwise, log warnings and return None
    Separately, if args or kwargs are given, format cache_dir using kwargs
    """
    cache_dir = cache_dir.format(*args, **kwargs)
    if path.isabs(cache_dir):
        if cache_root_dir is not None:
            logger.warning('cache_dir ({}) is a full path; ignoring cache_root_dir'.format(cache_dir))
        return cache_dir
    if cache_root_dir is not None:
        return path.join(cache_root_dir, cache_dir)
    else:
        logger.warning("cache dir is not full path & cache_root_dir not given. Caching may not work as expected!")
    return None