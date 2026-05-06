def _cache(method):
    """
    Decorator for caching data source return values

    Create a key index for the proxied array in the context.
    Iterate over the array shape descriptor e.g. (ntime, nbl, 3)
    returning tuples containing the lower and upper extents
    of string dimensions. Takes (0, d) in the case of an integer
    dimensions.
    """

    @functools.wraps(method)
    def memoizer(self, context):
        # Construct the key for the given index
        idx = context.array_extents(context.name)
        key = tuple(i for t in idx for i in t)

        with self._lock:
            # Access the sub-cache for this data source
            array_cache = self._cache[context.name]

            # Cache miss, call the data source
            if key not in array_cache:
                array_cache[key] = method(context)

            return array_cache[key]

    return memoizer