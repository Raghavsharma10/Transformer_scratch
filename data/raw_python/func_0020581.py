def top_by_func(self, sort_func):
        """
        Return home/away by player info for the players on each team who come in first according to the
        provided sorting function. Will perform ascending sort.
        
        :param sort_func: function that yields the sorting quantity
        :returns: dict of the form ``{ 'home/away': { by_player_dict } }``. See :py:func:`home_players` and :py:func:`away_players`
        """
        res = self.sort_players(sort_func=sort_func, reverse=True)
        return {
            'home': res['home'][0],
            'away': res['away'][0]
        }