def put(self, json=None):
        """Send a PUT request and return the JSON decoded result.

        Args:
            json (dict, optional): Object to encode and send in request.

        Returns:
            mixed: JSON decoded response data.
        """
        return self._call('put', url=self.endpoint, json=json)