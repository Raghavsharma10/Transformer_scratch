def by_zone(self):
        """
        Returns the faceoff win/total breakdown by zone for home and away as
        
        .. code:: python
        
            { 'home/away': {
                'off/def/neut/all': { 'won': won, 'total': total }
                }
            }
            
        :returns: dict
        """
        if self.__team_tots is None:
            self.__team_tots = self.__comp_tot()
        
        return {
            t: {
                z: self.__team_tots[t][z]
                for z in self.__zones
                if z != 'all'
            }
            for t in [ 'home', 'away' ]
        }