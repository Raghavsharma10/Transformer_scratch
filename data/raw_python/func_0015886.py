def post(self, url, data, headers=None):
        """
        Perform an HTTP POST request for a given url.
        Returns the response object.
        """
        return self._request('POST', url, data, headers=headers)