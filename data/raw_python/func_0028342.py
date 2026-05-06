def fetch_identifier_component(self, endpoint_name, identifier_data, query_params=None):
        """Common method for handling parameters before passing to api_client"""

        if query_params is None:
            query_params = {}

        identifier_input = self.get_identifier_input(identifier_data)

        return self._api_client.fetch(endpoint_name, identifier_input, query_params)