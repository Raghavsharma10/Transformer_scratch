def request(self, method, url, **kwargs):
        """Constructs a :class:`requests.Request`, prepares it and sends it.
        Raises HTTPErrors by default.

        :param method: method for the new :class:`Request` object.
        :type method: :class:`str`
        :param url: URL for the new :class:`Request` object.
        :type url: :class:`str`
        :param kwargs: keyword arguments of :meth:`requests.Session.request`
        :returns: a resonse object
        :rtype: :class:`requests.Response`
        :raises: :class:`requests.HTTPError`
        """
        if oauthlib.oauth2.is_secure_transport(url):
            m = super(OAuthSession, self).request
        else:
            m = super(requests_oauthlib.OAuth2Session, self).request
        log.debug("%s \"%s\" with %s", method, url, kwargs)
        response = m(method, url, **kwargs)
        response.raise_for_status()
        return response