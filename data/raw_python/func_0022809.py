def lighter(self, dv=0.1, copy=True):
        """Produce a lighter color (if possible)

        Parameters
        ----------
        dv : float
            Amount to increase the color value by.
        copy : bool
            If False, operation will be carried out in-place.

        Returns
        -------
        color : instance of ColorArray
            The lightened Color.
        """
        color = self.copy() if copy else self
        color.value += dv
        return color