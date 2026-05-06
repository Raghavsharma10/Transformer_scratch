def collections(self, values):
        """Set list of collections."""
        # if cache server is configured, save collection list
        if self.cache:
            self.cache.set(
                self.app.config['COLLECTIONS_CACHE_KEY'], values)