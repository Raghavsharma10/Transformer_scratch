def to_fits(self, filename, wavelengths=None, flux_unit=None, area=None,
                vegaspec=None, **kwargs):
        """Write the spectrum to a FITS file.

        Parameters
        ----------
        filename : str
            Output filename.

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        flux_unit : str or `~astropy.units.core.Unit` or `None`
            Flux is converted to this unit before written out.
            If not given, internal unit is used.

        area, vegaspec
            See :func:`~synphot.units.convert_flux`.

        kwargs : dict
            Keywords accepted by :func:`~synphot.specio.write_fits_spec`.

        """
        w, y = self._get_arrays(wavelengths, flux_unit=flux_unit, area=area,
                                vegaspec=vegaspec)

        # There are some standard keywords that should be added
        # to the extension header.
        bkeys = {'tdisp1': 'G15.7', 'tdisp2': 'G15.7'}

        if 'expr' in self.meta:
            bkeys['expr'] = (self.meta['expr'], 'synphot expression')

        if 'ext_header' in kwargs:
            kwargs['ext_header'].update(bkeys)
        else:
            kwargs['ext_header'] = bkeys

        specio.write_fits_spec(filename, w, y, **kwargs)