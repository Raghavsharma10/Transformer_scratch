def make_pdb(self):
        """Generates a PDB string for the `Monomer`."""
        pdb_str = write_pdb(
            [self], ' ' if not self.parent else self.parent.id)
        return pdb_str