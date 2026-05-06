def sort_players(self, sort_key=None, sort_func=None, reverse=False):
        """
        Return all home and away by player info sorted by either the provided key or function. Must provide
        at least one of the two parameters. Can sort either ascending or descending.
        
        :param sort_key: (def None) dict key to sort on
        :param sort_func: (def None) sorting function
        :param reverse: (optional, def False) if True, sort descending
        :returns: dict of the form ``{ 'home/away': { by_player_dict } }``. See :py:func:`home_players` and :py:func:`away_players`
        """
        def each(d):
            t = [ ]
            for num, v in d.items():
                ti = { vk: vv for vk, vv in v.items() }
                ti['num'] = num
                t.append(ti)
            
            if sort_key:
                return sorted(t, key=lambda k: k[sort_key], reverse=reverse)
            else:
                return sorted(t, key=sort_func, reverse=reverse)
            
        return self.__apply_to_both(each)