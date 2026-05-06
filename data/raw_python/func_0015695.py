def cache_return(func):
    """Cache the return value of a function without arguments"""

    _cache = []

    def wrap():
        if not _cache:
            _cache.append(func())
        return _cache[0]
    return wrap