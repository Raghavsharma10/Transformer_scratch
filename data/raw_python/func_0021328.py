def _get_cache_plus_key(self):
        """Return a cache region plus key."""
        key = getattr(self, '_cache_key', self.key_from_query())
        return self._cache.cache, key