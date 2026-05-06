def constant_cache(method):
    """
    Caches constant arrays associated with an array name.

    The intent of this decorator is to avoid the cost
    of recreating and storing many arrays of constant data,
    especially data created by np.zeros or np.ones.
    Instead, a single array of the first given shape is created
    and any further requests for constant data of the same
    (or smaller) shape are served from the cache.

    Requests for larger shapes or different types are regarded
    as a cache miss and will result in replacement of the
    existing cache value.
    """
    @functools.wraps(method)
    def wrapper(self, context):
        # Defer to method if no caching is enabled
        if not self._is_cached:
            return method(self, context)

        name = context.name
        cached = self._constant_cache.get(name, None)

        # No cached value, call method and return
        if cached is None:
            data = self._constant_cache[name] = method(self, context)
            return data

        # Can we just slice the existing cache entry?
        # 1. Are all context.shape's entries less than or equal
        #    to the shape of the cached data?
        # 2. Do they have the same dtype?
        cached_ok = (cached.dtype == context.dtype and
            all(l <= r for l,r in zip(context.shape, cached.shape)))

        # Need to return something bigger or a different type
        if not cached_ok:
            data = self._constant_cache[name] = method(self, context)
            return data

        # Otherwise slice the cached data
        return cached[tuple(slice(0, s) for s in context.shape)]

    f = wrapper
    f.__decorator__ = constant_cache.__name__

    return f