def all(self, query=None):
        """
        Gets resource collection for _resource_class.
        """

        if query is None:
            query = {}
        return self.client._get(
            self._url(),
            query
        )