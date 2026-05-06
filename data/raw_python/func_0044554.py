def query_parameters(self):
        """
        A key to list of values mapping of the query parameters seen in the
        request.
        """
        result = getattr(self, "_query_parameters", None)
        if result is None:
            result = self._query_parameters = normalize_query_parameters(
                self.query_string)
        return result