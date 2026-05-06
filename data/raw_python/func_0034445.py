def mk_url(self, *args, **kwargs):
        """
        Args get parameterized into base url:
            *(foo, bar, baz) -> /foo/bar/baz
        Kwargs get encoded and appended to base url:
            **{'hello':'world'} -> /foo/bar/baz?hello=world
        """
        params = urlencode(kwargs)
        url = '/' + '/'.join([x for x in args if x])
        if params:
            url += '?' + params
        return url