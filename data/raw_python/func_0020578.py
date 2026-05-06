def filter_players(self, pl_filter):
        """
        Return the subset home and away players that satisfy the provided filter function.
        
        :param pl_filter: function that takes a by player dictionary and returns bool
        :returns: dict of the form ``{ 'home/away': { by_player_dict } }``. See :py:func:`home_players` and :py:func:`away_players`
        """
        def each(d):
            return {
                k: v
                for k, v in d.items()
                if pl_filter(k, v)
            }
            
        return self.__apply_to_both(each)