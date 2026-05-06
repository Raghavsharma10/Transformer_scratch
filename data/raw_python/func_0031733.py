def oldapi_request(self, method, endpoint, **kwargs):
        """Make a request to one of the old api endpoints.

        The url will be constructed of :data:`TWITCH_APIURL` and
        the given endpoint.

        :param method: the request method
        :type method: :class:`str`
        :param endpoint: the endpoint of the old api.
                         The base url is automatically provided.
        :type endpoint: :class:`str`
        :param kwargs: keyword arguments of :meth:`requests.Session.request`
        :returns: a resonse object
        :rtype: :class:`requests.Response`
        :raises: :class:`requests.HTTPError`
        """
        headers = kwargs.setdefault('headers', {})
        headers['Client-ID'] = CLIENT_ID  # https://github.com/justintv/Twitch-API#rate-limits
        url = TWITCH_APIURL + endpoint
        return self.request(method, url, **kwargs)