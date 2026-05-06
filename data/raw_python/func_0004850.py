def yticktext(self, labels, index=1):
        """Set the tick labels.

        Parameters
        ----------
        labels : array-like

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['ticktext'] = labels
        return self