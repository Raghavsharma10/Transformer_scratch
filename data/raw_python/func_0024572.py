def center_atoms(self):
        """ get list of atoms of reaction center (atoms with dynamic: bonds, charges, radicals).
        """
        nodes = set()
        for n, atom in self.atoms():
            if atom._reactant != atom._product:
                nodes.add(n)

        for n, m, bond in self.bonds():
            if bond._reactant != bond._product:
                nodes.add(n)
                nodes.add(m)

        return list(nodes)