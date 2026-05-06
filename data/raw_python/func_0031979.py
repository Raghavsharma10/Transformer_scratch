def wrap_topgames(cls, response):
        """Wrap the response from quering the top games into instances
        and return them

        :param response: The response for quering the top games
        :type response: :class:`requests.Response`
        :returns: the new game instances
        :rtype: :class:`list` of :class:`Game`
        :raises: None
        """
        games = []
        json = response.json()
        topjsons = json['top']
        for t in topjsons:
            g = cls.wrap_json(json=t['game'],
                              viewers=t['viewers'],
                              channels=t['channels'])
            games.append(g)
        return games