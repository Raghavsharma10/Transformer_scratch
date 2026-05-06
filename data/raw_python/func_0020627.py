def fo_pct_by_zone(self):
        """
        Get the by team face-off win % by zone. Format is
            
        :returns: dict ``{ 'home/away': { 'off/def/neut': % } }``
        """
        bz = self.by_zone
        return {
            t: {
                z: bz[t][z]['won']/(1.0*bz[t][z]['total']) if bz[t][z]['total'] else 0.0
                for z in self.__zones
                if z != 'all'
            }
            for t in [ 'home', 'away' ]
            
        }