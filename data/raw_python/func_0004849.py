def ytickvals(self, values, index=1):
        """Set the tick values.

        Parameters
        ----------
        values : array-like

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['tickvals'] = values
        return self