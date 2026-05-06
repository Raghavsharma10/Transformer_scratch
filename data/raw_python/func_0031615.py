def to_pybel(self):
        """
        Produce a Pybel Molecule object.

        It is based on the capabilities of OpenBabel through Pybel. The present
        object must have at least `atomcoords`, `atomnos`, `charge` and `mult`
        defined.

        Returns
        -------
        `pybel.Molecule`

        Examples
        --------
        >>> from pyrrole.atoms import Atoms
        >>> dioxygen = Atoms({'atomcoords': [[0., 0., 0.],
        ...                                  [0., 0., 1.21]],
        ...                   'atomnos': [8, 8],
        ...                   'charge': 0,
        ...                   'mult': 3,
        ...                   'name': 'dioxygen'})
        >>> mol = dioxygen.to_pybel()
        >>> mol.molwt
        31.9988

        """
        # TODO: This only exports last geometry by default.
        obmol = _makeopenbabel(self.atomcoords, self.atomnos, self.charge,
                               self.mult)

        title = self.name or ""
        if 'scfenergies' in self.attributes:
            title += ", scfenergy={} eV".format(self.scfenergies[-1])
        obmol.SetTitle(title)

        # TODO: make a test for this function.
        return _pb.Molecule(obmol)