def get_first_molecule(self):
        """Get the first molecule from the trajectory

           This can be useful to configure your program before handeling the
           actual trajectory.
        """
        title, coordinates = self._first
        molecule = Molecule(self.numbers, coordinates, title, symbols=self.symbols)
        return molecule