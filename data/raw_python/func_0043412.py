def clean_cache(cached, **kwargs):
    " Generate cache key and clean cached value. "

    if isinstance(cached, basestring):
        cached = _str_to_model(cached)

    cache_key = generate_cache_key(cached, **kwargs)
    cache.delete(cache_key)