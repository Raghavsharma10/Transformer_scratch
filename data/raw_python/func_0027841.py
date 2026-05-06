def cache(self, key, value):
        """
        Add an entry to the cache.

        A weakref to the value is stored, rather than a direct reference. The
        value must have a C{__finalizer__} method that returns a callable which
        will be invoked when the weakref is broken.

        @param key: The key identifying the cache entry.

        @param value: The value for the cache entry.
        """
        fin = value.__finalizer__()
        try:
            # It's okay if there's already a cache entry for this key as long
            # as the weakref has already been broken. See the comment in
            # get() for an explanation of why this might happen.
            if self.data[key]() is not None:
                raise CacheInconsistency(
                    "Duplicate cache key: %r %r %r" % (
                        key, value, self.data[key]))
        except KeyError:
            pass
        callback = createCacheRemoveCallback(self._ref(self), key, fin)
        self.data[key] = self._ref(value, callback)
        return value