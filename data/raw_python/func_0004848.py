def yticksize(self, size, index=1):
        """Set the tick font size.

        Parameters
        ----------
        size : int

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['tickfont']['size'] = size
        return self