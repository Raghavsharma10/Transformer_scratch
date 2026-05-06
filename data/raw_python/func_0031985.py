def wrap_get_stream(cls, response):
        """Wrap the response from getting a stream into an instance
        and return it

        :param response: The response from getting a stream
        :type response: :class:`requests.Response`
        :returns: the new stream instance
        :rtype: :class:`list` of :class:`stream`
        :raises: None
        """
        json = response.json()
        s = cls.wrap_json(json['stream'])
        return s