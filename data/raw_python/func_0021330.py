def set_value(self, value):
        """Set the value in the cache for this query."""
        cache, cache_key = self._get_cache_plus_key()
        cache.set(cache_key, value)