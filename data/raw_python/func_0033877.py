def _request(self, uri, params=None, postParams=None, verb='GET'):
        """Execute a request on the plugit api"""
        return getattr(requests, verb.lower())(self.url + uri, params=params, data=postParams, stream=True)