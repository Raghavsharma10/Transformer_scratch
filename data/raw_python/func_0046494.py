def delete_many(self, uris):
        """
        Remove many nodes from cache.
        No return.
        """
        cache_keys = (self._build_cache_key(uri) for uri in uris)
        self._delete_many(cache_keys)