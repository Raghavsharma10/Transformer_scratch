def put(self, endpoint, **kwargs):
        """Update a resource.

        Args:
            endpoint: resource endpoint.
        """
        return self._request(requests.put, endpoint, **kwargs)