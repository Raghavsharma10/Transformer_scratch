def delete_connections(self, **kwargs):
        """Remove a single connection to a provider for the specified user."""
        rv = False
        for c in self.find_connections(**kwargs):
            self.delete(c)
            rv = True
        return rv