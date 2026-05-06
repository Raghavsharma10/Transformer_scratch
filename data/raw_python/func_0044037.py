def close_monomers(self, group, cutoff=4.0):
        """Returns a list of Monomers from within a cut off distance of the Monomer

        Parameters
        ----------
        group: BaseAmpal or Subclass
            Group to be search for Monomers that are close to this Monomer.
        cutoff: float
            Distance cut off.

        Returns
        -------
        nearby_residues: [Monomers]
            List of Monomers within cut off distance.
        """
        nearby_residues = []
        for self_atom in self.atoms.values():
            nearby_atoms = group.is_within(cutoff, self_atom)
            for res_atom in nearby_atoms:
                if res_atom.parent not in nearby_residues:
                    nearby_residues.append(res_atom.parent)
        return nearby_residues