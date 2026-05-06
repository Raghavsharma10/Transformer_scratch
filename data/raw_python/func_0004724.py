def get(self, endpoint, **kwargs):
        """Get a resource.

        Args:
            endpoint: resource endpoint.
        """
        return self._request(requests.get, endpoint, **kwargs)