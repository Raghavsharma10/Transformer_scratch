def append(self, traverse):
        """(traverse)->append the traverse to the current traverse"""
        self.data.extend(traverse.data)
        self.atoms.extend(traverse.atoms)
        self.bonds.extend(traverse.bonds)