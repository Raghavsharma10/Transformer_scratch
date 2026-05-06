def _put(self, url, attributes=None, **kwargs):
        """
        Wrapper for the HTTP PUT request.
        """

        return self._request('put', url, attributes, **kwargs)