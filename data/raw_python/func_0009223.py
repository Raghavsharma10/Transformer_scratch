def post(self, path, query=None, data=None, redirects=True):
        """
        POST request wrapper for :func:`request()`
        """
        return self.request('POST', path, query, data, redirects)