def fo_pct(self):
        """
        Get the by team overall face-off win %.
        
        :returns: dict, ``{ 'home': %, 'away': % }``
        """
        tots = self.team_totals
        return {
            t: tots[t]['won']/(1.0*tots[t]['total']) if tots[t]['total'] else 0.0
            for t in [ 'home', 'away' ]
        }