def post(self, endpoint, **kwargs):
        """Create a resource.

        Args:
            endpoint: resource endpoint.
        """
        return self._request(requests.post, endpoint, **kwargs)