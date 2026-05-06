def ytickangle(self, angle, index=1):
        """Set the angle of the y-axis tick labels.

        Parameters
        ----------
        value : int
            Angle in degrees
        index : int, optional
            Y-axis index

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['tickangle'] = angle
        return self