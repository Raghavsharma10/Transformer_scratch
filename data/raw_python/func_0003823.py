def from_molecule(cls, molecule, labels=None):
        """Initialize a similarity descriptor

           Arguments:
             molecule  --  a Molecules object
             labels  --  a list with integer labels used to identify atoms of
                         the same type. When not given, the atom numbers from
                         the molecule are used.
        """
        if labels is None:
            labels = molecule.numbers
        return cls(molecule.distance_matrix, labels)