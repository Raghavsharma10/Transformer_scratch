def to_funset(self, discrete):
        """
        Converts the dataset to a set of `gringo.Fun`_ instances

        Parameters
        ----------
        discrete : callable
            A discretization function

        Returns
        -------
        set
            Representation of the dataset as a set of `gringo.Fun`_ instances


        .. _gringo.Fun: http://potassco.sourceforge.net/gringo.html#Fun
        """
        fs = self.clampings.to_funset("exp")
        fs = fs.union(self.setup.to_funset())

        for i, row in self.readouts.iterrows():
            for var, val in row.iteritems():
                if not np.isnan(val):
                    fs.add(gringo.Fun('obs', [i, var, discrete(val)]))

        return fs