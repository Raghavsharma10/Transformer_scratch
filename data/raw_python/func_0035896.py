def get(self, params=None):
        """Send a POST request and return the JSON decoded result.

        Args:
            params (dict, optional): Mapping of parameters to send in request.

        Returns:
            mixed: JSON decoded response data.
        """
        return self._call('get', url=self.endpoint, params=params)