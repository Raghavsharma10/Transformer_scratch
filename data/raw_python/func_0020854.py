def sets(self, values):
        """Set list of sets."""
        # if cache server is configured, save sets list
        if self.cache:
            self.cache.set(self.app.config['OAISERVER_CACHE_KEY'], values)