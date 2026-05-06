def find(self, resource_id, query=None, **kwargs):
        """Gets a single resource."""

        if query is None:
            query = {}
        return self.client._get(
            self._url(resource_id),
            query,
            **kwargs
        )