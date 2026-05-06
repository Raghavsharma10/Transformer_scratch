def get_cached_output(self, placeholder_name, instance):
        """
        .. versionadded:: 0.9
           Return the cached output for a rendered item, or ``None`` if no output is cached.

           This method can be overwritten to implement custom caching mechanisms.
           By default, this function generates the cache key using :func:`get_output_cache_key`
           and retrieves the results from the configured Django cache backend (e.g. memcached).
        """
        cachekey = self.get_output_cache_key(placeholder_name, instance)
        return cache.get(cachekey)