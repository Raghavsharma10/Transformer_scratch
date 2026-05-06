def binned_pixelrange(self, waverange, **kwargs):
        """Calculate the number of pixels within the given wavelength
        range and `binset`.

        Parameters
        ----------
        waverange : tuple of float or `~astropy.units.quantity.Quantity`
            Lower and upper limits of the desired wavelength range.
            If not a Quantity, assumed to be in Angstrom.

        kwargs : dict
            Keywords accepted by :func:`synphot.binning.pixel_range`.

        Returns
        -------
        npix : int
            Number of pixels.

        """
        x = units.validate_quantity(
            waverange, self._internal_wave_unit, equivalencies=u.spectral())
        return binning.pixel_range(self.binset.value, x.value, **kwargs)