def post(self, *args, **kwargs):
        """Sends a POST request to a reddit path determined by ``args``.  Basically ``.post('foo', 'bar', 'baz')`` will POST http://www.reddit.com/foo/bar/baz/.json.  ``kwargs`` supplied will be passed to ``requests.post`` after having ``modhash`` and ``cookies`` injected, and after having modhash injected into ``kwargs['data']`` if logged in.  Injection only occurs if they don't already exist.
        
        Returns received response JSON content as a dict.
        
        Raises :class:`exceptions.BadResponse` if not a 200 response or no JSON content received or raises :class:`exceptions.PostError` if a reddit error was returned.
        
        :param \*args: strings that will form the path to POST
        :param \*\*kwargs: extra keyword arguments to be passed to ``requests.POST``
        """
        kwargs = self._inject_request_kwargs(kwargs)
        kwargs = self._inject_post_data(kwargs)
        url = reddit_url(*args)
        r = requests.post(url, **kwargs)
        if r.status_code == 200:
            try:
                j = json.loads(r.content)
            except ValueError:
                raise BadResponse(r)
            try:
                errors = j['json']['errors']
            except Exception:
                errors = None
            if errors:
                raise PostError(errors)
            else:
                return j
        else:
            raise BadResponse(r)