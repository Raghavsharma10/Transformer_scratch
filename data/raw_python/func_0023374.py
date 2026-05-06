def map(self, x):
        """The Python mapping function from the [0,1] interval to a
        list of rgba colors

        Parameters
        ----------
        x : array-like
            The values to map.

        Returns
        -------
        colors : list
            List of rgba colors.
        """
        return self._map_function(self.colors.rgba, x, self._controls)