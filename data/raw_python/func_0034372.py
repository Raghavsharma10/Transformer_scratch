def limit(self, value):
        """
        Allows for limiting number of results returned for query. Useful
        for pagination.
        """
        self._query = self._query.limit(value)

        return self