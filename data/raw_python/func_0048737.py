def is_valid_query(self, query):
        """
        Return True if the search query is valid.

        e.g.:
        * not empty,
        * not too short,
        """
        # No query, no item
        if not query:
            return False
        # Query is too short, no item
        if len(query) < self.get_query_size_min():
            return False
        return True