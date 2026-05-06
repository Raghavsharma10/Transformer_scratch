def wrap_get_channel(cls, response):
        """Wrap the response from getting a channel into an instance
        and return it

        :param response: The response from getting a channel
        :type response: :class:`requests.Response`
        :returns: the new channel instance
        :rtype: :class:`list` of :class:`channel`
        :raises: None
        """
        json = response.json()
        c = cls.wrap_json(json)
        return c