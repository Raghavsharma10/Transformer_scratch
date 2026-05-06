def top_games(self, limit=10, offset=0):
        """Return the current top games

        :param limit: the maximum amount of top games to query
        :type limit: :class:`int`
        :param offset: the offset in the top games
        :type offset: :class:`int`
        :returns: a list of top games
        :rtype: :class:`list` of :class:`models.Game`
        :raises: None
        """
        r = self.kraken_request('GET', 'games/top',
                                params={'limit': limit,
                                        'offset': offset})
        return models.Game.wrap_topgames(r)