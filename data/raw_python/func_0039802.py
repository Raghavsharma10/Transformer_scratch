def get(self, *args, **kwargs):
        """Sends a GET request to a reddit path determined by ``args``.  Basically ``.get('foo', 'bar', 'baz')`` will GET http://www.reddit.com/foo/bar/baz/.json.  ``kwargs`` supplied will be passed to :meth:`requests.get` after having ``user_agent`` and ``cookies`` injected.  Injection only occurs if they don't already exist.
        
        Returns :class:`things.Blob` object or a subclass of :class:`things.Blob`, or raises :class:`exceptions.BadResponse` if not a 200 Response.
        
        :param \*args: strings that will form the path to GET
        :param \*\*kwargs: extra keyword arguments to be passed to :meth:`requests.get`
        """
        kwargs = self._inject_request_kwargs(kwargs)
        url = reddit_url(*args)
        r = requests.get(url, **kwargs)
        # print r.url
        if r.status_code == 200:
            thing = self._thingify(json.loads(r.content), path=urlparse(r.url).path)
            return thing
        else:
            raise BadResponse(r)