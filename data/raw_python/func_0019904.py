def post_parse(self, uri, params={}, data={}):
        '''Convenience method to call post() on an arbitrary URI and parse the response
        into a JSON object. Raises an error on non-200 response status.
        '''
        return self._request_parse(self.post, uri, params, data)