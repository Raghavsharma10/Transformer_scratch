def unique_id(self):
        """Creates a unique ID for the `Atom` based on its parents.

        Returns
        -------
        unique_id : (str, str, str)
            (polymer.id, residue.id, atom.id)
        """
        chain = self.parent.parent.id
        residue = self.parent.id
        return chain, residue, self.id