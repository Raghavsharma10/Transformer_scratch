def wrap_search(cls, response):
        """Wrap the response from a game search into instances
        and return them

        :param response: The response from searching a game
        :type response: :class:`requests.Response`
        :returns: the new game instances
        :rtype: :class:`list` of :class:`Game`
        :raises: None
        """
        games = []
        json = response.json()
        gamejsons = json['games']
        for j in gamejsons:
            g = cls.wrap_json(j)
            games.append(g)
        return games