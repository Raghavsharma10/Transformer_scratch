def get_game(self, name):
        """Get the game instance for a game name

        :param name: the name of the game
        :type name: :class:`str`
        :returns: the game instance
        :rtype: :class:`models.Game` | None
        :raises: None
        """
        games = self.search_games(query=name, live=False)
        for g in games:
            if g.name == name:
                return g