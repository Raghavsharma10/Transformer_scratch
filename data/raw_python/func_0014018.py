def _app_cache_deepcopy(obj):
    """
    An helper that correctly deepcopy model cache state
    """
    if isinstance(obj, defaultdict):
        return deepcopy(obj)
    elif isinstance(obj, dict):
        return type(obj)((_app_cache_deepcopy(key), _app_cache_deepcopy(val)) for key, val in obj.items())
    elif isinstance(obj, list):
        return list(_app_cache_deepcopy(val) for val in obj)
    elif isinstance(obj, AppConfig):
        app_conf = Empty()
        app_conf.__class__ = AppConfig
        app_conf.__dict__ = _app_cache_deepcopy(obj.__dict__)
        return app_conf
    return obj