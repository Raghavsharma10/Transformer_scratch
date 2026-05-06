def ylim(self, low, high, index=1):
        """Set yaxis limits.

        Parameters
        ----------
        low : number
        high : number
        index : int, optional

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['range'] = [low, high]
        return self