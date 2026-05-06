def from_str(cls, string):
        """
        Creates a mapping from a string

        Parameters
        ----------
        string : str
            String of the form `target<-clause` where `clause` is a valid string for :class:`caspo.core.clause.Clause`

        Returns
        -------
        caspo.core.mapping.Mapping
            Created object instance
        """
        if "<-" not in string:
            raise ValueError("Cannot parse the given string to a mapping")

        target, clause_str = string.split('<-')

        return cls(Clause.from_str(clause_str), target)