def store(self, key, expiry, data):
        """
        Add a result to the cache

        :key: Cache key to use
        :expiry: The expiry timestamp after which the result is stale
        :data: The data to cache
        """
        self.cache.set(key, (expiry, data), self.cache_ttl)

        if getattr(settings, 'CACHEBACK_VERIFY_CACHE_WRITE', True):
            # We verify that the item was cached correctly.  This is to avoid a
            # Memcache problem where some values aren't cached correctly
            # without warning.
            __, cached_data = self.cache.get(key, (None, None))
            if data is not None and cached_data is None:
                raise RuntimeError(
                    "Unable to save data of type %s to cache" % (
                        type(data)))