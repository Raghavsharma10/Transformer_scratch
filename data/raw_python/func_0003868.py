def rmsd(self, other):
        """Compute the RMSD between two molecules.

           Arguments:
            | ``other``  --  Another molecule with the same atom numbers

           Return values:
            | ``transformation``  --  the transformation that brings 'self' into
                                  overlap with 'other'
            | ``other_trans``  --  the transformed coordinates of geometry 'other'
            | ``rmsd``  --  the rmsd of the distances between corresponding atoms in
                            'self' and 'other'

           Make sure the atoms in `self` and `other` are in the same order.

           Usage::

             >>> print molecule1.rmsd(molecule2)[2]/angstrom
        """
        if self.numbers.shape != other.numbers.shape or \
           (self.numbers != other.numbers).all():
            raise ValueError("The other molecule does not have the same numbers as this molecule.")
        return fit_rmsd(self.coordinates, other.coordinates)