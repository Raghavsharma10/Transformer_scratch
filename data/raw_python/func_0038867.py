def get_cacheable(cache_key, cache_ttl, calculate, recalculate=False):
    """
    Gets the result of a method call, using the given key and TTL as a cache
    """
    if not recalculate:
        cached = cache.get(cache_key)
        if cached is not None:
            return json.loads(cached)

    calculated = calculate()
    cache.set(cache_key, json.dumps(calculated), cache_ttl)

    return calculated