def has_cache(self):
        """Intended to be called before any call that might access the
        cache. If the cache is not selected, then returns False,
        otherwise the cache is build if needed and returns True."""
        if not self.cache_enabled:
            return False
        if self._cache is None:
            self.build_cache()
        return True