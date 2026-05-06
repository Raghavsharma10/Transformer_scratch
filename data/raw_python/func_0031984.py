def wrap_search(cls, response):
        """Wrap the response from a stream search into instances
        and return them

        :param response: The response from searching a stream
        :type response: :class:`requests.Response`
        :returns: the new stream instances
        :rtype: :class:`list` of :class:`stream`
        :raises: None
        """
        streams = []
        json = response.json()
        streamjsons = json['streams']
        for j in streamjsons:
            s = cls.wrap_json(j)
            streams.append(s)
        return streams