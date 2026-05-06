def from_file(cls, filename, **kwargs):
        """Create a reddening law from file.

        If filename has 'fits' or 'fit' suffix, it is read as FITS.
        Otherwise, it is read as ASCII.

        Parameters
        ----------
        filename : str
            Reddening law filename.

        kwargs : dict
            Keywords acceptable by
            :func:`~synphot.specio.read_fits_spec` (if FITS) or
            :func:`~synphot.specio.read_ascii_spec` (if ASCII).

        Returns
        -------
        redlaw : `ReddeningLaw`
            Empirical reddening law.

        """
        if 'flux_unit' not in kwargs:
            kwargs['flux_unit'] = cls._internal_flux_unit

        if ((filename.endswith('fits') or filename.endswith('fit')) and
                'flux_col' not in kwargs):
            kwargs['flux_col'] = 'Av/E(B-V)'

        header, wavelengths, rvs = specio.read_spec(filename, **kwargs)

        return cls(Empirical1D, points=wavelengths, lookup_table=rvs,
                   meta={'header': header})