def center_bonds(self):
        """ get list of bonds of reaction center (bonds with dynamic orders).
        """
        return [(n, m) for n, m, bond in self.bonds() if bond._reactant != bond._product]