def copy(self, newdata=None):
        '''Return a copy of the cube with optionally new data.'''
        if newdata is None:
            newdata = self.data.copy()
        return self.__class__(
            self.molecule, self.origin.copy(), self.axes.copy(),
            self.nrep.copy(), newdata, self.subtitle, self.nuclear_charges
        )