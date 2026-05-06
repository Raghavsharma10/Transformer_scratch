def authenticated_request(self, endpoint, method='GET', params=None, data=None):
        '''
        Send a request to the given Wunderlist API with 'X-Access-Token' and 'X-Client-ID' headers and ensure the response code is as expected given the request type

        Params:
        endpoint -- API endpoint to send request to

        Keyword Args:
        method -- GET, PUT, PATCH, DELETE, etc.
        params -- parameters to encode in the request
        data -- data to send with the request
        '''
        headers = {
                'X-Access-Token' : self.access_token,
                'X-Client-ID' : self.client_id
                }
        return self.api.request(endpoint, method=method, headers=headers, params=params, data=data)