def get_optimized_molecule(self):
        """Return a molecule object of the optimal geometry"""
        opt_coor = self.get_optimization_coordinates()
        if len(opt_coor) == 0:
            return None
        else:
            return Molecule(
                self.molecule.numbers,
                opt_coor[-1],
            )