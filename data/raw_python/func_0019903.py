def get_parse(self, uri, params={}):
        '''Convenience method to call get() on an arbitrary URI and parse the response
        into a JSON object. Raises an error on non-200 response status.
        '''
        return self._request_parse(self.get, uri, params)