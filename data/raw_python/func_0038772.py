def drop_literals(self, literals):
        """
        Returns a new list of clampings without the given literals

        Parameters
        ----------
        literals : iterable[:class:`caspo.core.literal.Literal`]
            Iterable of literals to be removed from each clamping


        Returns
        -------
        caspo.core.clamping.ClampingList
            The new list of clampings
        """
        clampings = []
        for clamping in self:
            c = clamping.drop_literals(literals)
            if len(c) > 0:
                clampings.append(c)

        return ClampingList(clampings)