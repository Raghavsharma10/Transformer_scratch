def fetch_viewers(self, game):
        """Query the viewers and channels of the given game and
        set them on the object

        :returns: the given game
        :rtype: :class:`models.Game`
        :raises: None
        """
        r = self.kraken_request('GET', 'streams/summary',
                                params={'game': game.name}).json()
        game.viewers = r['viewers']
        game.channels = r['channels']
        return game