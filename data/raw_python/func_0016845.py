def chunk_cache(method):
    """
    Caches chunks of default data.

    This decorator caches generated default data so as to
    avoid recomputing it on a subsequent queries to the
    provider.
    """

    @functools.wraps(method)
    def wrapper(self, context):
        # Defer to the method if no caching is enabled
        if not self._is_cached:
            return method(self, context)

        # Construct the key for the given index
        name = context.name
        idx = context.array_extents(name)
        key = tuple(i for t in idx for i in t)
        # Access the sub-cache for this array
        array_cache = self._chunk_cache[name]

        # Cache miss, call the function
        if key not in array_cache:
            array_cache[key] = method(self, context)

        return array_cache[key]

    f = wrapper
    f.__decorator__ = chunk_cache.__name__
    return f