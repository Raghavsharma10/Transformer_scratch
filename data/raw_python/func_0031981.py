def wrap_search(cls, response):
        """Wrap the response from a channel search into instances
        and return them

        :param response: The response from searching a channel
        :type response: :class:`requests.Response`
        :returns: the new channel instances
        :rtype: :class:`list` of :class:`channel`
        :raises: None
        """
        channels = []
        json = response.json()
        channeljsons = json['channels']
        for j in channeljsons:
            c = cls.wrap_json(j)
            channels.append(c)
        return channels