def _call_api_many_related_resources(self, resource, url_list,
                                         method_name, **kwargs):
        """
        For HypermediaResource - make an API call to a list of known URLs
        """
        responses = []
        for url in url_list:
            params = {
                'headers': self.get_http_headers(
                    resource.Meta.name, method_name, **kwargs),
                'url': url
            }
            prepared_request = self.prepare_http_request(
                'GET', params, **kwargs)
            response = self.session.send(prepared_request)
            result = self._handle_response(
                response, resource.Meta.valid_status_codes, resource)
            if len(result) > 1:
                responses.append(result)
            else:
                responses.append(result[0])
        return responses