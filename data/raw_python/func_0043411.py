def cached_query(qs, timeout=None):
    """ Auto cached queryset and generate results.
    """
    cache_key = generate_cache_key(qs)
    return get_cached(cache_key, list, args=(qs,), timeout=None)