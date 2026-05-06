def top_by_key(self, sort_key):
        """
        Return home/away by player info for the players on each team that are first in the provided category.
        
        :param sort_key: str, the dictionary key to be sorted on
        :returns: dict of the form ``{ 'home/away': { by_player_dict } }``. See :py:func:`home_players` and :py:func:`away_players`
        """
        res = self.sort_players(sort_key=sort_key, reverse=True)
        return {
            'home': res['home'][0],
            'away': res['away'][0]
        }