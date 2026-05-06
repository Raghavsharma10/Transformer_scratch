def fetch_coords(self, query):
        """Pull down coordinate data from the endpoint."""
        q = query.add_query_parameter(req='coord')
        return self._parse_messages(self.get_query(q).content)