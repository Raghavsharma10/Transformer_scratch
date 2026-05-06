def validate_query(self, query):
        """Confirm query exists given common filters."""
        if query is None:
            return query
        query = self.update_reading_list(query)
        return query