def prepare_http_request(self, method_type, params, **kwargs):
        """
        Prepares the HTTP REQUEST and returns it.

        Args:
            method_type: The HTTP method type
            params: Additional parameters for the HTTP request.
            kwargs: Any extra keyword arguements passed into a client method.

        returns:
            prepared_request: An HTTP request object.
        """
        prepared_request = self.session.prepare_request(
            requests.Request(method=method_type, **params)
        )
        return prepared_request