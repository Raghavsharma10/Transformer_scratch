def cache_if_needed(cacheKey, result, menu, context, meta):
    """Cache the result, if needed"""

    if cacheKey:

        # This will be a method in django 1.7
        flat_context = {}
        for d in context.dicts:
            flat_context.update(d)

        del flat_context['csrf_token']

        data = {'result': result, 'menu': menu, 'context': flat_context}

        cache.set('plugit-cache-' + cacheKey, data, meta['cache_time'])