def from_file(cls, filename, keep_neg=False, **kwargs):
        """Create a spectrum from file.

        If filename has 'fits' or 'fit' suffix, it is read as FITS.
        Otherwise, it is read as ASCII.

        Parameters
        ----------
        filename : str
            Spectrum filename.

        keep_neg : bool
            See `~synphot.models.Empirical1D`.

        kwargs : dict
            Keywords acceptable by
            :func:`~synphot.specio.read_fits_spec` (if FITS) or
            :func:`~synphot.specio.read_ascii_spec` (if ASCII).

        Returns
        -------
        sp : `SourceSpectrum`
            Empirical spectrum.

        """
        header, wavelengths, fluxes = specio.read_spec(filename, **kwargs)
        return cls(Empirical1D, points=wavelengths, lookup_table=fluxes,
                   keep_neg=keep_neg, meta={'header': header})