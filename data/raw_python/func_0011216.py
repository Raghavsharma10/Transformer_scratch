def request_encode_url(self, method, url, fields=None, **urlopen_kw):
        """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the url. This is useful for request methods like GET, HEAD, DELETE, etc.
        """
        if fields:
            url += '?' + urlencode(fields)
        return self.urlopen(method, url, **urlopen_kw)