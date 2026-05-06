def from_str(cls, string):
        """
        Creates a literal from a string

        Parameters
        ----------
        string : str
            If the string starts with '!', it's interpreted as a negated variable

        Returns
        -------
        caspo.core.literal.Literal
            Created object instance
        """
        if string[0] == '!':
            signature = -1
            variable = string[1:]
        else:
            signature = 1
            variable = string

        return cls(variable, signature)