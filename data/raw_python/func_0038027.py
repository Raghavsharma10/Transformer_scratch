def from_str(cls, string):
        """
        Creates a clause from a given string.

        Parameters
        ----------
        string: str
             A string of the form `a+!b` which translates to `a AND NOT b`.

        Returns
        -------
        caspo.core.clause.Clause
            Created object instance
        """
        return cls([Literal.from_str(lit) for lit in string.split('+')])