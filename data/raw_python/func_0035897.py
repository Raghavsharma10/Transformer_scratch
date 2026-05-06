def post(self, json=None):
        """Send a POST request and return the JSON decoded result.

        Args:
            json (dict, optional): Object to encode and send in request.

        Returns:
            mixed: JSON decoded response data.
        """
        return self._call('post', url=self.endpoint, json=json)