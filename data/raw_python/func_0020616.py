def ColMap(season):
        """
        Returns a dictionary mapping the type of information in the RTSS play row to the
        appropriate column number. The column locations pre/post 2008 are different.
        
        :param season: int for the season number
        :returns: mapping of RTSS column to info type
        :rtype: dict, keys are ``'play_num', 'per', 'str', 'time', 'event', 'desc', 'vis', 'home'``
        """
        if c.MIN_SEASON <= season <= c.MAX_SEASON:
            return {
                "play_num": 0,
                "per": 1,
                "str": 2,
                "time": 3,
                "event": 4,
                "desc": 5,
                "vis": 6,
                "home": 7
            }
        else:
            raise ValueError("RTSSCol.MAP(season): Invalid season " + str(season))