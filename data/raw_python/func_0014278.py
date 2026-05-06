def from_pdb(cls, path, forcefield=None, loader=PDBFile, strict=True, **kwargs):
        """
        Loads topology, positions and, potentially, velocities and vectors,
        from a PDB or PDBx file

        Parameters
        ----------
        path : str
            Path to PDB/PDBx file
        forcefields : list of str
            Paths to FFXML and/or FRCMOD forcefields. REQUIRED.

        Returns
        -------
        pdb : SystemHandler
            SystemHandler with topology, positions, and, potentially, velocities and
            box vectors. Forcefields are embedded in the `master` attribute.
        """
        pdb = loader(path)
        box = kwargs.pop('box', pdb.topology.getPeriodicBoxVectors())
        positions = kwargs.pop('positions', pdb.positions)
        velocities = kwargs.pop('velocities', getattr(pdb, 'velocities', None))

        if strict and not forcefield:
            from .md import FORCEFIELDS as forcefield
            logger.info('! Forcefields for PDB not specified. Using default: %s',
                        ', '.join(forcefield))
        pdb.forcefield = ForceField(*list(process_forcefield(*forcefield)))

        return cls(master=pdb.forcefield, topology=pdb.topology, positions=positions,
                   velocities=velocities, box=box, path=path, **kwargs)