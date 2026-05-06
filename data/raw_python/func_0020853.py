def sets(self):
        """Get list of sets."""
        if self.cache:
            return self.cache.get(
                self.app.config['OAISERVER_CACHE_KEY'])