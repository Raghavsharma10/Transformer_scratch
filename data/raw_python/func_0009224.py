def get(self, path, query=None, redirects=True):
        """
        GET request wrapper for :func:`request()`
        """
        return self.request('GET', path, query, None, redirects)