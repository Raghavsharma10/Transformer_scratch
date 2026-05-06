def from_file(cls, filename, temperature_key='DEFT',
                  beamfill_key='BEAMFILL', **kwargs):
        """Creates a thermal spectral element from file.

        .. note::

            Only FITS format is supported.

        Parameters
        ----------
        filename : str
            Thermal spectral element filename.

        temperature_key, beamfill_key : str
            Keywords in FITS *table extension* that store temperature
            (in Kelvin) and beam filling factor values.
            Beam filling factor is set to 1 if its keyword is missing.

        kwargs : dict
            Keywords acceptable by :func:`~synphot.specio.read_fits_spec`.

        Returns
        -------
        th : `ThermalSpectralElement`
            Empirical thermal spectral element.

        Raises
        ------
        synphot.exceptions.SynphotError
            Invalid inputs.

        """
        if not (filename.endswith('fits') or filename.endswith('fit')):
            raise exceptions.SynphotError('Only FITS format is supported.')

        # Extra info from table header
        ext = kwargs.get('ext', 1)
        tab_hdr = fits.getheader(filename, ext=ext)

        temperature = tab_hdr.get(temperature_key)
        if temperature is None:
            raise exceptions.SynphotError(
                'Missing {0} keyword.'.format(temperature_key))

        beam_fill_factor = tab_hdr.get('BEAMFILL', 1)

        if 'flux_unit' not in kwargs:
            kwargs['flux_unit'] = cls._internal_flux_unit

        if 'flux_col' not in kwargs:
            kwargs['flux_col'] = 'EMISSIVITY'

        header, wavelengths, em = specio.read_spec(filename, **kwargs)
        return cls(
            Empirical1D, temperature, beam_fill_factor=beam_fill_factor,
            points=wavelengths, lookup_table=em, meta={'header': header})