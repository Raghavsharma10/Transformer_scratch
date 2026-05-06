def from_mapping(cls, x0, x1):
        """ Create an STTransform from the given mapping

        See `set_mapping` for details.

        Parameters
        ----------
        x0 : array-like
            Start.
        x1 : array-like
            End.

        Returns
        -------
        t : instance of STTransform
            The transform.
        """
        t = cls()
        t.set_mapping(x0, x1)
        return t