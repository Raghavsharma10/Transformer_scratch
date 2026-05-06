def request(self, endpoint, method='GET', headers=None, params=None, data=None):
        '''
        Send a request to the given Wunderlist API endpoint

        Params:
        endpoint -- API endpoint to send request to

        Keyword Args:
        headers -- headers to add to the request
        method -- GET, PUT, PATCH, DELETE, etc.
        params -- parameters to encode in the request
        data -- data to send with the request
        '''
        if not headers:
            headers = {}
        if method in ['POST', 'PATCH', 'PUT']:
            headers['Content-Type'] = 'application/json'
        url = '/'.join([self.api_url, 'v' + self.api_version, endpoint])
        data = json.dumps(data) if data else None
        try:
            response = requests.request(method=method, url=url, params=params, headers=headers, data=data)
        # TODO Does recreating the exception classes 'requests' use suck? Yes, but it sucks more to expose the underlying library I use
        except requests.exceptions.Timeout as e:
            raise wp_exceptions.TimeoutError(e)
        except requests.exceptions.ConnectionError as e:
            raise wp_exceptions.ConnectionError(e)
        self._validate_response(method, response)
        return response