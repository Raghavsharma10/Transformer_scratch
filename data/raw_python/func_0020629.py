def share(self):
        """
        The Cori-share (% of shot attempts) for each team
        
        :returns: dict, ``{ 'home_name': %, 'away_name': % }``
        """
        tot = sum(self.total.values())
        return { k: v/float(tot) for k,v in self.total.items() }