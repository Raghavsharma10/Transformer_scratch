def head_to_head(self, home_num, away_num):
        """
        Return the head-to-head face-off outcomes between two players.
        If the matchup didn't happen, ``{ }`` is returned.
        
        :param home_num: the number of the home team player
        :param away_num: the number of the away team player
        :returns: dict, either ``{ }`` or the following
        
        .. code:: python
        
            {
                'home/away': {
                    'off/def/neut/all': { 'won': won, 'total': total }
                }
            }
        """
        if home_num in self.home_fo and away_num in self.home_fo[home_num]['opps']:
            h_fo = self.home_fo[home_num]['opps'][away_num]
            a_fo = self.away_fo[away_num]['opps'][home_num]
            return {
                'home': { k: h_fo[k] for k in self.__zones },
                'away': { k: a_fo[k] for k in self.__zones }
            }
        else:
            return { }