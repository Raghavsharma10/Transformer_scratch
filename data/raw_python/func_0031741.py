def get_streams(self, game=None, channels=None, limit=25, offset=0):
        """Return a list of streams queried by a number of parameters
        sorted by number of viewers descending

        :param game: the game or name of the game
        :type game: :class:`str` | :class:`models.Game`
        :param channels: list of models.Channels or channel names (can be mixed)
        :type channels: :class:`list` of :class:`models.Channel` or :class:`str`
        :param limit: maximum number of results
        :type limit: :class:`int`
        :param offset: offset for pagination
        :type offset: :class:`int`
        :returns: A list of streams
        :rtype: :class:`list` of :class:`models.Stream`
        :raises: None
        """
        if isinstance(game, models.Game):
            game = game.name

        channelnames = []
        cparam = None
        if channels:
            for c in channels:
                if isinstance(c, models.Channel):
                    c = c.name
                channelnames.append(c)
            cparam = ','.join(channelnames)

        params = {'limit': limit,
                  'offset': offset,
                  'game': game,
                  'channel': cparam}

        r = self.kraken_request('GET', 'streams', params=params)
        return models.Stream.wrap_search(r)