def _approximate_unkown_bond_lengths(self):
        """Completes the bond length database with approximations based on VDW radii"""
        dataset = self.lengths[BOND_SINGLE]
        for n1 in periodic.iter_numbers():
            for n2 in periodic.iter_numbers():
                if n1 <= n2:
                    pair = frozenset([n1, n2])
                    atom1 = periodic[n1]
                    atom2 = periodic[n2]
                    #if (pair not in dataset) and hasattr(atom1, "covalent_radius") and hasattr(atom2, "covalent_radius"):
                    if (pair not in dataset) and (atom1.covalent_radius is not None) and (atom2.covalent_radius is not None):
                        dataset[pair] = (atom1.covalent_radius + atom2.covalent_radius)