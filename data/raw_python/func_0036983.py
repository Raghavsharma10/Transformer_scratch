def _call_api_single_related_resource(self, resource, full_resource_url,
                                          method_name, **kwargs):
        """
        For HypermediaResource - make an API call to a known URL
        """
        url = full_resource_url
        params = {
            'headers': self.get_http_headers(
                resource.Meta.name, method_name, **kwargs),
            'url': url
        }
        prepared_request = self.prepare_http_request(
            'GET', params, **kwargs)
        response = self.session.send(prepared_request)
        return self._handle_response(
            response, resource.Meta.valid_status_codes, resource)