def from_amber(cls, path, positions=None, strict=True, **kwargs):
        """
        Loads Amber Parm7 parameters and topology file

        Parameters
        ----------
        path : str
            Path to *.prmtop or *.top file
        positions : simtk.unit.Quantity
            Atomic positions

        Returns
        -------
        prmtop : SystemHandler
            SystemHandler with topology
        """
        if strict and positions is None:
            raise ValueError('Amber TOP/PRMTOP files require initial positions.')
        prmtop = AmberPrmtopFile(path)
        box = kwargs.pop('box', prmtop.topology.getPeriodicBoxVectors())
        return cls(master=prmtop, topology=prmtop.topology, positions=positions, box=box,
                   path=path, **kwargs)