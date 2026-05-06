def ylabelsize(self, size, index=1):
        """Set the size of the label.

        Parameters
        ----------
        size : int

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['titlefont']['size'] = size
        return self