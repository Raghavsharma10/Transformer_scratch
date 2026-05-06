def cache_result(cache_key, timeout):
    """A decorator for caching the result of a function."""
    def decorator(f):
        cache_name = settings.WAFER_CACHE

        @functools.wraps(f)
        def wrapper(*args, **kw):
            cache = caches[cache_name]
            result = cache.get(cache_key)
            if result is None:
                result = f(*args, **kw)
                cache.set(cache_key, result, timeout)
            return result

        def invalidate():
            cache = caches[cache_name]
            cache.delete(cache_key)

        wrapper.invalidate = invalidate
        return wrapper
    return decorator