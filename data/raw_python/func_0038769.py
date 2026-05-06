def frequency(self, literal):
        """
        Returns the frequency of a clamped variable

        Parameters
        ----------
        literal : :class:`caspo.core.literal.Literal`
            The clamped variable

        Returns
        -------
        float
            The frequency of the given literal

        Raises
        ------
        ValueError
            If the variable is not present in any of the clampings
        """
        df = self.to_dataframe()
        if literal.variable in df.columns:
            return len(df[df[literal.variable] == literal.signature]) / float(len(self))
        else:
            raise ValueError("Variable not found: %s" % literal.variable)