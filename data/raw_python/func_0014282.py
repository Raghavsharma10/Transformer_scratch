def from_gromacs(cls, path, positions=None, forcefield=None, strict=True, **kwargs):
        """
        Loads a topology from a Gromacs TOP file located at `path`.

        Additional root directory for parameters can be specified with `forcefield`.

        Arguments
        ---------
        path : str
            Path to a Gromacs TOP file
        positions : simtk.unit.Quantity
            Atomic positions
        forcefield : str, optional
            Root directory for parameter files
        """
        if strict and positions is None:
            raise ValueError('Gromacs TOP files require initial positions.')
        box = kwargs.pop('box', None)
        top = GromacsTopFile(path, includeDir=forcefield, periodicBoxVectors=box)
        return cls(master=top, topology=top.topology, positions=positions, box=box,
                   path=path, **kwargs)