def to_funset(self, lname="clamping", cname="clamped"):
        """
        Converts the list of clampings to a set of `gringo.Fun`_ instances

        Parameters
        ----------
        lname : str
            Predicate name for the clamping id

        cname : str
            Predicate name for the clamped variable

        Returns
        -------
        set
            Representation of all clampings as a set of `gringo.Fun`_ instances


        .. _gringo.Fun: http://potassco.sourceforge.net/gringo.html#Fun
        """
        fs = set()
        for i, clamping in enumerate(self):
            fs.add(gringo.Fun(lname, [i]))
            fs = fs.union(clamping.to_funset(i, cname))

        return fs