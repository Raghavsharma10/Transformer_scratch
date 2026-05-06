def from_charmm(cls, path, positions=None, forcefield=None, strict=True, **kwargs):
        """
        Loads PSF Charmm structure from `path`. Requires `charmm_parameters`.

        Parameters
        ----------
        path : str
            Path to PSF file
        forcefield : list of str
            Paths to Charmm parameters files, such as *.par or *.str. REQUIRED

        Returns
        -------
        psf : SystemHandler
            SystemHandler with topology. Charmm parameters are embedded in
            the `master` attribute.
        """
        psf = CharmmPsfFile(path)
        if strict and forcefield is None:
            raise ValueError('PSF files require key `forcefield`.')
        if strict and positions is None:
            raise ValueError('PSF files require key `positions`.')
        psf.parmset = CharmmParameterSet(*forcefield)
        psf.loadParameters(psf.parmset)
        return cls(master=psf, topology=psf.topology, positions=positions, path=path,
                   **kwargs)