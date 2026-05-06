def darker(self, dv=0.1, copy=True):
        """Produce a darker color (if possible)

        Parameters
        ----------
        dv : float
            Amount to decrease the color value by.
        copy : bool
            If False, operation will be carried out in-place.

        Returns
        -------
        color : instance of ColorArray
            The darkened Color.
        """
        color = self.copy() if copy else self
        color.value -= dv
        return color