def call_api(self, method_type, method_name,
                 valid_status_codes, resource, data,
                 uid, **kwargs):
        """
        Make HTTP calls.

        Args:
            method_type: The HTTP method
            method_name: The name of the python method making the HTTP call
            valid_status_codes: A tuple of integer status codes
                                deemed acceptable as response statuses
            resource: The resource class that will be generated
            data: The post data being sent.
            uid: The unique identifier of the resource.
        Returns:

        kwargs is a list of keyword arguments. Additional custom keyword
        arguments can be sent into this method and will be passed into
        subclass methods:

        - get_url
        - prepare_http_request
        - get_http_headers
        """
        url = resource.get_resource_url(
            resource, base_url=self.Meta.base_url
        )
        if method_type in SINGLE_RESOURCE_METHODS:
            if not uid and not kwargs:
                raise MissingUidException
            url = resource.get_url(
                url=url, uid=uid, **kwargs)
        params = {
            'headers': self.get_http_headers(
                self.Meta.name, method_name, **kwargs),
            'url': url
        }
        if method_type in ['POST', 'PUT', 'PATCH'] and isinstance(data, dict):
            params.update(json=data)
        prepared_request = self.prepare_http_request(
            method_type, params, **kwargs)
        response = self.session.send(prepared_request)
        return self._handle_response(response, valid_status_codes, resource)