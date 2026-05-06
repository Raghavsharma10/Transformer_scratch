def delete(self, uri):
        """
        Remove node uri from cache.
        No return.
        """
        cache_key = self._build_cache_key(uri)
        self._delete(cache_key)