def _get(self, url, query=None, **kwargs):
        """
        Wrapper for the HTTP GET request.
        """

        return self._request('get', url, query, **kwargs)