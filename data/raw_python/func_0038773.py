def to_funset(self, index, name="clamped"):
        """
        Converts the clamping to a set of `gringo.Fun`_ object instances

        Parameters
        ----------
        index : int
            An external identifier to associate several clampings together in ASP

        name : str
            A function name for the clamping

        Returns
        -------
        set
            The set of `gringo.Fun`_ object instances


        .. _gringo.Fun: http://potassco.sourceforge.net/gringo.html#Fun
        """
        fs = set()
        for var, sign in self:
            fs.add(gringo.Fun(name, [index, var, sign]))

        return fs