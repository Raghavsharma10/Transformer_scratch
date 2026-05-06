def usher_request(self, method, endpoint, **kwargs):
        """Make a request to one of the usher api endpoints.

        The url will be constructed of :data:`TWITCH_USHERURL` and
        the given endpoint.

        :param method: the request method
        :type method: :class:`str`
        :param endpoint: the endpoint of the usher api.
                         The base url is automatically provided.
        :type endpoint: :class:`str`
        :param kwargs: keyword arguments of :meth:`requests.Session.request`
        :returns: a resonse object
        :rtype: :class:`requests.Response`
        :raises: :class:`requests.HTTPError`
        """
        url = TWITCH_USHERURL + endpoint
        return self.request(method, url, **kwargs)