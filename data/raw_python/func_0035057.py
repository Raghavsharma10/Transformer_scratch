def rotate(self, atom):
        """(atom)->start the cycle at position atom, assumes
        that atom is in the cycle"""
        try:
            index = self.atoms.index(atom)
        except ValueError:
            raise CycleError("atom %s not in cycle"%(atom))

        self.atoms = self.atoms[index:] + self.atoms[:index]
        self.bonds = self.bonds[index:] + self.bonds[:index]