def find_in_cache(cacheKey):
    """Check if the content exists in cache and return it"""
    # If we have to use cache, we try to find the result in cache
    if cacheKey:

        data = cache.get('plugit-cache-' + cacheKey, None)

        # We found a result, we can return it
        if data:
            return (data['result'], data['menu'], data['context'])
    return (None, None, None)