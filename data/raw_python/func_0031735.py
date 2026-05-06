def search_games(self, query, live=True):
        """Search for games that are similar to the query

        :param query: the query string
        :type query: :class:`str`
        :param live: If true, only returns games that are live on at least one
                     channel
        :type live: :class:`bool`
        :returns: A list of games
        :rtype: :class:`list` of :class:`models.Game` instances
        :raises: None
        """
        r = self.kraken_request('GET', 'search/games',
                                params={'query': query,
                                        'type': 'suggest',
                                        'live': live})
        games = models.Game.wrap_search(r)
        for g in games:
            self.fetch_viewers(g)
        return games