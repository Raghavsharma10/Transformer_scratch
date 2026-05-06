def api_post(self, action, data, binary_data_param=None):
        """
        Perform an HTTP POST request, using the shared-secret auth hash.
        @param action: API action call
        @param data: dictionary values
        """
        binary_data_param = binary_data_param or []
        if binary_data_param:
            return self.api_post_multipart(action, data, binary_data_param)
        else:
            return self._api_request(action, data, 'POST')