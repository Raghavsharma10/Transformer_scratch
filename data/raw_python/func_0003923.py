def get_molecule(self, index=0):
        """Get a molecule from the trajectory

           Optional argument:
            | ``index``  --  The frame index [default=0]
        """
        return Molecule(self.numbers, self.geometries[index], self.titles[index], symbols=self.symbols)