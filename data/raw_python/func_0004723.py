def _request(self, method, resource_uri, **kwargs):
        """Perform a method on a resource.

        Args:
            method: requests.`method`
            resource_uri: resource endpoint
        Raises:
            HTTPError
        Returns:
            JSON Response
        """
        data = kwargs.get('data')
        response = method(self.API_BASE_URL + resource_uri,
                          json=data, headers=self.headers)
        response.raise_for_status()
        return response.json()