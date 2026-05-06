def delete(self, json=None):
        """Send a DELETE request and return the JSON decoded result.

        Args:
            json (dict, optional): Object to encode and send in request.

        Returns:
            mixed: JSON decoded response data.
        """
        return self._call('delete', url=self.endpoint, json=json)