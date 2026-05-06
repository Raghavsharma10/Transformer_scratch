def from_url(cls, url, show_host=True):
        '''Parse string and get URL instance'''
        # url must be idna-encoded and url-quotted

        if six.PY2:
            if isinstance(url, six.text_type):
                url = url.encode('utf-8')
            parsed = urlparse(url)
            netloc = parsed.netloc.decode('utf-8') # XXX HACK
        else:# pragma: no cover
            if isinstance(url, six.binary_type):
                url = url.decode('utf-8', errors='replace') # XXX
            parsed = urlparse(url)
            netloc = parsed.netloc

        query = _parse_qs(parsed.query)
        host = netloc.split(':', 1)[0] if ':' in netloc else netloc

        port = netloc.split(':')[1] if ':' in netloc else ''
        path = unquote(parsed.path)
        fragment = unquote(parsed.fragment)
        if not fragment and not url.endswith('#'):
            fragment = None
        return cls(path,
                   query, host,
                   port, parsed.scheme, fragment, show_host)