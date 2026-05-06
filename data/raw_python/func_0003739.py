def get_length(self, n1, n2, bond_type=BOND_SINGLE):
        """Return the length of a bond between n1 and n2 of type bond_type

           Arguments:
            | ``n1``  --  the atom number of the first atom in the bond
            | ``n2``  --  the atom number of the second atom the bond

           Optional argument:
            | ``bond_type``  --  the type of bond [default=BOND_SINGLE]

           This is a safe method for querying a bond_length. If no answer can be
           found, this get_length returns None.
        """
        dataset = self.lengths.get(bond_type)
        if dataset == None:
            return None
        return dataset.get(frozenset([n1, n2]))