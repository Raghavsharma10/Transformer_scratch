def offset(self, value):
        """
        Allows for skipping a specified number of results in query. Useful
        for pagination.
        """

        self._query = self._query.skip(value)

        return self