def centroid(self):
        """Calculates the centroid of the residue.

        Returns
        -------
        centroid : numpy.array or None
            Returns a 3D coordinate for the residue unless a CB
            atom is not available, in which case `None` is
            returned.

        Notes
        -----
        Uses the definition of the centroid from Huang *et al* [2]_.

        References
        ----------
        .. [2] Huang ES, Subbiah S and Levitt M (1995) Recognizing Native
           Folds by the Arrangement of Hydrophobic and Polar Residues, J. Mol.
           Biol return., **252**, 709-720.
        """
        if 'CB' in self.atoms:
            cb_unit_vector = unit_vector(
                self['CB']._vector - self['CA']._vector)
            return self['CA']._vector + (3.0 * cb_unit_vector)
        return None