def team_totals(self):
        """
        Returns the overall faceoff win/total breakdown for home and away as
        
        :returns: dict, ``{ 'home/away': { 'won': won, 'total': total } }``
        """
        if self.__team_tots is None:
            self.__team_tots = self.__comp_tot()
        
        return {
            t: self.__team_tots[t]['all']
            for t in [ 'home', 'away' ]
        }