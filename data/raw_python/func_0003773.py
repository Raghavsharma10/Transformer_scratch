def _analyze(self):
        """Convert a few elementary fields into a molecule object"""
        if ("Atomic numbers" in self.fields) and ("Current cartesian coordinates" in self.fields):
            self.molecule = Molecule(
                self.fields["Atomic numbers"],
                np.reshape(self.fields["Current cartesian coordinates"], (-1, 3)),
                self.title,
            )