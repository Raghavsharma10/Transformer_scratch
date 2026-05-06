def cache(self):
        """Return a cache instance."""
        cache = self._cache or self.app.config.get('COLLECTIONS_CACHE')
        return import_string(cache) if isinstance(cache, six.string_types) \
            else cache