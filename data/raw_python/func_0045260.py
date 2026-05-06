def cached_unless_authenticated(timeout=50, key_prefix='default'):
    """Cache anonymous traffic."""
    def caching(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            cache_fun = current_cache.cached(
                timeout=timeout, key_prefix=key_prefix,
                unless=lambda: current_cache_ext.is_authenticated_callback())
            return cache_fun(f)(*args, **kwargs)
        return wrapper
    return caching