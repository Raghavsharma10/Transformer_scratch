def api_get(self, action, data, headers=None):
        """
        Perform an HTTP GET request, using the shared-secret auth hash.
        @param action: API action call
        @param data: dictionary values
        """
        return self._api_request(action, data, 'GET', headers)