def xatom(self, atom):
        """(atom)->return the atom at the other end of this bond
        or None if atom is not part of this bond"""
        handle = atom.handle
        
        if handle == self.atoms[0].handle:
            return self.atoms[1]
        elif handle == self.atoms[1].handle:
            return self.atoms[0]
        return None