def pruneToAtoms(self, atoms):
        """Prune the molecule to the specified atoms
        bonds will be removed atomatically"""
        _atoms = self.atoms[:]
        for atom in _atoms:
            if atom not in atoms:
                self.remove_atom(atom)