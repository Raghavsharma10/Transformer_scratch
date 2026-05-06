def cache(self):
        """Query or return the Graph API representation of this resource."""
        if not self._cache:
            self._cache = self.graph.get('%s' % self.id)

        return self._cache