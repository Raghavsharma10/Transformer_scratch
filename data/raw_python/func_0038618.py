def to_funset(self):
        """
        Converts the experimental setup to a set of `gringo.Fun`_ object instances

        Returns
        -------
        set
            The set of `gringo.Fun`_ object instances


        .. _gringo.Fun: http://potassco.sourceforge.net/gringo.html#Fun
        """
        fs = set((gringo.Fun('stimulus', [str(var)]) for var in self.stimuli))
        fs = fs.union((gringo.Fun('inhibitor', [str(var)]) for var in self.inhibitors))
        fs = fs.union((gringo.Fun('readout', [str(var)]) for var in self.readouts))

        return fs