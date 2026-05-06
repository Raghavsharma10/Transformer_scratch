def distance_restraint_force(self, atoms, distances, strengths):
        """
        Parameters
        ----------
        atoms : tuple of tuple of int or str
            Pair of atom indices to be restrained, with shape (n, 2),
            like ((a1, a2), (a3, a4)). Items can be str compatible with MDTraj DSL.
        distances : tuple of float
            Equilibrium distances for each pair
        strengths : tuple of float
            Force constant for each pair
        """
        system = self.system
        force = mm.HarmonicBondForce()
        force.setUsesPeriodicBoundaryConditions(self.system.usesPeriodicBoundaryConditions())
        for pair, distance, strength in zip(atoms, distances, strengths):
            indices = []
            for atom in pair:
                if isinstance(atom, str):
                    index = self.subset(atom)
                    if len(index) != 1:
                        raise ValueError('Distance restraint for selection `{}` returns != 1 atom!: {}'
                                         .format(atom, index))
                    indices.append(int(index[0]))
                elif isinstance(atom, (int, float)):
                    indices.append(int(atom))
                else:
                    raise ValueError('Distance restraint atoms must be int or str DSL selections')
            if distance == 'current':
                pos = self.positions or system.positions
                distance = np.linalg.norm(pos[indices[0]] - pos[indices[1]])

            force.addBond(indices[0], indices[1], distance*u.nanometers,
                          strength*u.kilocalories_per_mole/u.angstroms**2)
        return force