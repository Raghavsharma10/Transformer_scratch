def to_string(self, format="smi", dialect=None, with_header=False,
                  fragment_id=None, constraints=None):
        r"""
        Produce a string representation of the molecule.

        This function wraps and extends the functionality of OpenBabel (which
        is accessible through `to_pybel`). Many chemical formats can thus be
        output (see the `pybel.outformats` variable for a list of available
        output formats).

        Parameters
        ----------
        format : `str`, optional
            Chemical file format of the returned string representation (see
            examples below).
        dialect : `str`, optional
            Format dialect. This encompasses enhancements provided for some
            subformats. If ``"standard"`` or `None`, the output provided by
            OpenBabel is used with no or minimal modification. See notes below.
        with_header : `bool`, optional
            If `format` encompasses a header, allow it in the returned string.
            This would be, for instance, the first two lines of data for
            ``format="xyz"`` (see examples below). This might not work with all
            dialects and/or formats.
        fragment_id : `str`, optional
            Indentify molecular fragments (see examples below). This might not
            work with all dialects and/or formats.
        constraints : iterable object of `int`
            Set cartesian constraints for selected atoms (see examples below).
            This might not work with all dialects and/or formats.

        Returns
        -------
        `str`
            String representation of molecule in the specified format and/or
            dialect.

        Raises
        ------
        KeyError
            Raised if `dialect` value is currently not supported or if
            `fragment_id` is given with a currently not supported `dialect`
            value.

        Notes
        -----
        Format dialects are subformats that support extended functionality.
        Currently supported dialects are:

        - for ``format="xyz"``:
            - ``"ADF"``, ``"ORCA"``.

        Examples
        --------
        >>> from pyrrole import atoms
        >>> dioxygen = atoms.Atoms({'atomcoords': [[0., 0., 0.],
        ...                                        [0., 0., 1.21]],
        ...                         'atomnos': [8, 8],
        ...                         'charge': 0,
        ...                         'mult': 3,
        ...                         'name': 'dioxygen'})

        By default, a SMILES string is returned:

        >>> dioxygen.to_string()
        'O=O\tdioxygen'

        Cartesian coordinates can be produced with ``format="xyz"``, which is
        equivalent to printing an `Atoms` instance:

        >>> print(dioxygen.to_string("xyz"))
        O          0.00000        0.00000        0.00000
        O          0.00000        0.00000        1.21000
        >>> print(dioxygen)
        O          0.00000        0.00000        0.00000
        O          0.00000        0.00000        1.21000

        Header lines are disabled by default (for ``format="xyz"``, for
        example, the header stores the number of atoms in the molecule and a
        comment or title line), but this can be reversed with
        ``with_header=True``:

        >>> print(dioxygen.to_string("xyz", with_header=True))
        2
        dioxygen
        O          0.00000        0.00000        0.00000
        O          0.00000        0.00000        1.21000

        Coordinates for packages such as GAMESS and MOPAC are also supported:

        >>> water_dimer = atoms.read_pybel("data/water-dimer.xyz")
        >>> print(water_dimer.to_string("gamin"))
        O      8.0     -1.6289300000   -0.0413800000    0.3713700000
        H      1.0     -0.6980300000   -0.0916800000    0.0933700000
        H      1.0     -2.0666300000   -0.7349800000   -0.1366300000
        O      8.0      1.2145700000    0.0317200000   -0.2762300000
        H      1.0      1.4492700000    0.9167200000   -0.5857300000
        H      1.0      1.7297700000   -0.0803800000    0.5338700000
        >>> print(water_dimer.to_string("mop"))
        O  -1.62893 1 -0.04138 1  0.37137 1
        H  -0.69803 1 -0.09168 1  0.09337 1
        H  -2.06663 1 -0.73498 1 -0.13663 1
        O   1.21457 1  0.03172 1 -0.27623 1
        H   1.44927 1  0.91672 1 -0.58573 1
        H   1.72977 1 -0.08038 1  0.53387 1

        Constraining of cartesian coordinates works with MOPAC format:

        >>> print(water_dimer.to_string("mop", constraints=(0, 3)))
        O  -1.62893 0 -0.04138 0  0.37137 0
        H  -0.69803 1 -0.09168 1  0.09337 1
        H  -2.06663 1 -0.73498 1 -0.13663 1
        O   1.21457 0  0.03172 0 -0.27623 0
        H   1.44927 1  0.91672 1 -0.58573 1
        H   1.72977 1 -0.08038 1  0.53387 1

        Fragment identification is supported for ``"ADF"`` and ``"ORCA"``
        dialects:

        >>> print(water_dimer.to_string("xyz", dialect="ADF",
        ...                             fragment_id="dimer"))
        O         -1.62893       -0.04138        0.37137       f=dimer
        H         -0.69803       -0.09168        0.09337       f=dimer
        H         -2.06663       -0.73498       -0.13663       f=dimer
        O          1.21457        0.03172       -0.27623       f=dimer
        H          1.44927        0.91672       -0.58573       f=dimer
        H          1.72977       -0.08038        0.53387       f=dimer
        >>> print(water_dimer.to_string("xyz", dialect="ORCA",
        ...                             fragment_id=1))
        O(1)      -1.62893       -0.04138        0.37137
        H(1)      -0.69803       -0.09168        0.09337
        H(1)      -2.06663       -0.73498       -0.13663
        O(1)       1.21457        0.03172       -0.27623
        H(1)       1.44927        0.91672       -0.58573
        H(1)       1.72977       -0.08038        0.53387

        """
        s = self.to_pybel().write(format).strip()

        if dialect is None:
            dialect = "standard"
        dialect = dialect.lower()

        if format == "xyz":
            natom, comment, body = s.split("\n", 2)

            if dialect in {"adf", "orca", "standard"}:
                if fragment_id is not None:
                    if dialect == "adf":
                        body = \
                            "\n".join(["{}       f={}".format(line,
                                                              fragment_id)
                                       for line in body.split("\n")])
                    elif dialect == "orca":
                        fragment_id = "({})".format(fragment_id)
                        body = \
                            "\n".join([line.replace(" " * len(fragment_id),
                                                    fragment_id, 1)
                                       for line in body.split("\n")])
                    else:
                        raise KeyError("fragment_id currently not supported "
                                       "with dialect '{}'".format(dialect))
            else:
                raise KeyError("dialect '{}' currently not "
                               "supported".format(dialect))

            if with_header:
                s = "\n".join([natom, comment, body])
            else:
                s = body

        elif format == "gamin":
            lines = s.split("\n")
            begin = "\n".join([line.strip() for line in lines[:5]])
            body = "\n".join([line.strip() for line in lines[5:-1]])

            if with_header:
                s = "\n".join([begin, body])
            else:
                s = body

        elif format == "mop":
            chunks = s.split("\n", 2)
            begin = "\n".join([line.strip() for line in chunks[:2]])
            body = chunks[2].strip()

            if constraints is not None:
                body = body.split("\n")
                for i in constraints:
                    body[i] = _re.sub(' 1( |$)', ' 0\g<1>', body[i])
                body = "\n".join(body)

            if with_header:
                s = "\n".join([begin, body])
            else:
                s = body

        return s.strip()