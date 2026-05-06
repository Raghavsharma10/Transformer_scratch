def createCacheRemoveCallback(cacheRef, key, finalizer):
    """
    Construct a callable to be used as a weakref callback for cache entries.

    The callable will invoke the provided finalizer, as well as removing the
    cache entry if the cache still exists and contains an entry for the given
    key.

    @type  cacheRef: L{weakref.ref} to L{FinalizingCache}
    @param cacheRef: A weakref to the cache in which the corresponding cache
        item was stored.

    @param key: The key for which this value is cached.

    @type  finalizer: callable taking 0 arguments
    @param finalizer: A user-provided callable that will be called when the
        weakref callback runs.
    """
    def remove(reference):
        # Weakref callbacks cannot raise exceptions or DOOM ensues
        try:
            finalizer()
        except:
            logErrorNoMatterWhat()
        try:
            cache = cacheRef()
            if cache is not None:
                if key in cache.data:
                    if cache.data[key] is reference:
                        del cache.data[key]
        except:
            logErrorNoMatterWhat()
    return remove