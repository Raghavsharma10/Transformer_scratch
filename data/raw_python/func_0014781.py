def _get_resource(self, relative_url, params=None):
        """Convenience function for retrieving a resource.
        If resource does not exist, return None.
        """
        response = self._get(relative_url, params=params, raise_for_status=False)
        if response.status_code == 404:
            return None
        self._raise_for_status(response)
        return response.json()