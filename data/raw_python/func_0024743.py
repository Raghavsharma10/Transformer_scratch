def to_fits(self, filename, wavelengths=None, **kwargs):
        """Write the reddening law to a FITS file.

        :math:`R(V)` column is automatically named 'Av/E(B-V)'.

        Parameters
        ----------
        filename : str
            Output filename.

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        kwargs : dict
            Keywords accepted by :func:`~synphot.specio.write_fits_spec`.

        """
        w, y = self._get_arrays(wavelengths)

        kwargs['flux_col'] = 'Av/E(B-V)'
        kwargs['flux_unit'] = self._internal_flux_unit

        # No need to trim/pad zeroes, unless user chooses to do so.
        if 'pad_zero_ends' not in kwargs:
            kwargs['pad_zero_ends'] = False
        if 'trim_zero' not in kwargs:
            kwargs['trim_zero'] = False

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