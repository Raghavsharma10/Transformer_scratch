def frequencies_iter(self):
        """
        Iterates over the frequencies of all clamped variables

        Yields
        ------
        tuple[ caspo.core.literal.Literal, float ]
            The next tuple of the form (literal, frequency)
        """
        df = self.to_dataframe()
        n = float(len(self))
        for var, sign in it.product(df.columns, [-1, 1]):
            f = len(df[df[var] == sign]) / n
            if f > 0:
                yield Literal(var, sign), f