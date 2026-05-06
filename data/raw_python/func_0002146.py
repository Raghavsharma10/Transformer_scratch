def fetch_header(self):
        """Make a header request to the endpoint."""
        query = self.query().add_query_parameter(req='header')
        return self._parse_messages(self.get_query(query).content)[0]