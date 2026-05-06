def split(self, pattern=None):
        r"""
        Break molecule up into constituent fragments.

        By default (i.e., if `pattern` is `None`), each disconnected fragment
        is returned as a separate new `Atoms` object. This uses OpenBabel
        (through `OBMol.Separate`) and might not preserve atom order, depending
        on your version of the library.

        Parameters
        ----------
        pattern : iterable of iterable of `int`, optional
            Groupings of atoms into molecule fragments. Each element of
            `pattern` should be an iterable whose members are atom indices (see
            example below).

        Returns
        -------
        fragments : iterable of `Atoms`

        Examples
        --------
        >>> from pyrrole import atoms
        >>> water_dimer = atoms.read_pybel("data/water-dimer.xyz")

        "Natural fragmentation" is the default behaviour, i.e. all disconnected
        fragments are returned:

        >>> for frag in water_dimer.split():
        ...     print("{}\n".format(frag))
        O         -1.62893       -0.04138        0.37137
        H         -0.69803       -0.09168        0.09337
        H         -2.06663       -0.73498       -0.13663
        <BLANKLINE>
        O          1.21457        0.03172       -0.27623
        H          1.72977       -0.08038        0.53387
        H          1.44927        0.91672       -0.58573
        <BLANKLINE>

        Precise fragment grouping can be achieved by explicitly indicating
        which atoms belong to which fragments:

        >>> for frag in water_dimer.split([range(3), (5, 4), [3]]):
        ...     print("{}\n".format(frag))
        O         -1.62893       -0.04138        0.37137
        H         -0.69803       -0.09168        0.09337
        H         -2.06663       -0.73498       -0.13663
        <BLANKLINE>
        H          1.72977       -0.08038        0.53387
        H          1.44927        0.91672       -0.58573
        <BLANKLINE>
        O          1.21457        0.03172       -0.27623
        <BLANKLINE>

        """
        molecule_pybel = self.to_pybel()

        if pattern is None:
            fragments = [read_pybel(frag)
                         for frag in molecule_pybel.OBMol.Separate()]
        else:
            fragments = []
            for group in pattern:
                fragment_obmol = _pb.ob.OBMol()
                for i in group:
                    obatom = molecule_pybel.OBMol.GetAtomById(i)
                    fragment_obmol.InsertAtom(obatom)

                fragments.append(fragment_obmol)

            fragments = [read_pybel(frag) for frag in fragments]

        return fragments