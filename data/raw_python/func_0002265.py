def set_cached_output(self, placeholder_name, instance, output):
        """
        .. versionadded:: 0.9
           Store the cached output for a rendered item.

           This method can be overwritten to implement custom caching mechanisms.
           By default, this function generates the cache key using :func:`~fluent_contents.cache.get_rendering_cache_key`
           and stores the results in the configured Django cache backend (e.g. memcached).

           When custom cache keys are used, also include those in :func:`get_output_cache_keys`
           so the cache will be cleared when needed.

        .. versionchanged:: 1.0
           The received data is no longer a HTML string, but :class:`~fluent_contents.models.ContentItemOutput` object.
        """
        cachekey = self.get_output_cache_key(placeholder_name, instance)
        if self.cache_timeout is not DEFAULT_TIMEOUT:
            cache.set(cachekey, output, self.cache_timeout)
        else:
            # Don't want to mix into the default 0/None issue.
            cache.set(cachekey, output)