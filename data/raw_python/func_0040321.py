def request(self, url, method = u"get", data = None, headers = None, **kwargs):
        """
        public method for doing the live request
        """

        url, method, data, headers, kwargs = self._pre_request(url, 
                                                                 method=method,
                                                                 data=data,
                                                                 headers=headers,
                                                                 **kwargs)
        response = self._request(url, method=method, data=data, headers=headers, **kwargs)
        response = self._post_request(response)
        
        # raises the appropriate exceptions
        response = self._handle_response(response)
        
        return response