def render_content(content, **kwargs):
    '''check content for "active" urls, store results to django cache'''
    # try to take pre rendered content from django cache, if caching is enabled
    if settings.ACTIVE_URL_CACHE:
        cache_key = get_cache_key(content, **kwargs)

        # get cached content from django cache backend
        from_cache = cache.get(cache_key)

        # return pre rendered content if it exist in cache
        if from_cache is not None:
            return from_cache

    # render content with "active" logic
    content = check_content(content, **kwargs)

    # write rendered content to django cache backend, if caching is enabled
    if settings.ACTIVE_URL_CACHE:
        cache.set(cache_key, content, settings.ACTIVE_URL_CACHE_TIMEOUT)

    return content