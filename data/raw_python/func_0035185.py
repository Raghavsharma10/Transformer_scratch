def getChirality(self, order):
        """(order)->what is the chirality of a given order of
        atoms?"""
        indices = tuple([self._initialOrder.index(atom.handle)
                         for atom in order])
        same = chiral_table[indices]
        if same:
            return self.chirality
        else:
            if self.chirality == "@": return "@@"
            else: return "@"