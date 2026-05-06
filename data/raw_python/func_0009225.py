def head(self, path, query=None, data=None, redirects=True):
        """
        HEAD request wrapper for :func:`request()`
        """
        return self.request('HEAD', path, query, None, redirects)