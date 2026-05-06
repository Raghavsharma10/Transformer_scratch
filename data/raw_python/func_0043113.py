def collections(self):
        """Get list of collections."""
        # if cache server is configured, load collection from there
        if self.cache:
            return self.cache.get(
                self.app.config['COLLECTIONS_CACHE_KEY'])