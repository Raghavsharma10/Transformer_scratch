def kraken_request(self, method, endpoint, **kwargs):
        """Make a request to one of the kraken api endpoints.

        Headers are automatically set to accept :data:`TWITCH_HEADER_ACCEPT`.
        Also the client id from :data:`CLIENT_ID` will be set.
        The url will be constructed of :data:`TWITCH_KRAKENURL` and
        the given endpoint.

        :param method: the request method
        :type method: :class:`str`
        :param endpoint: the endpoint of the kraken api.
                         The base url is automatically provided.
        :type endpoint: :class:`str`
        :param kwargs: keyword arguments of :meth:`requests.Session.request`
        :returns: a resonse object
        :rtype: :class:`requests.Response`
        :raises: :class:`requests.HTTPError`
        """
        url = TWITCH_KRAKENURL + endpoint
        headers = kwargs.setdefault('headers', {})
        headers['Accept'] = TWITCH_HEADER_ACCEPT
        headers['Client-ID'] = CLIENT_ID  # https://github.com/justintv/Twitch-API#rate-limits
        return self.request(method, url, **kwargs)