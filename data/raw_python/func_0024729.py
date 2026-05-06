def binned_waverange(self, cenwave, npix, **kwargs):
        """Calculate the wavelength range covered by the given number
        of pixels centered on the given central wavelengths of
        `binset`.

        Parameters
        ----------
        cenwave : float or `~astropy.units.quantity.Quantity`
            Desired central wavelength.
            If not a Quantity, assumed to be in Angstrom.

        npix : int
            Desired number of pixels, centered on ``cenwave``.

        kwargs : dict
            Keywords accepted by :func:`synphot.binning.wave_range`.

        Returns
        -------
        waverange : `~astropy.units.quantity.Quantity`
            Lower and upper limits of the wavelength range,
            in the unit of ``cenwave``.

        """
        # Calculation is done in the unit of cenwave.
        if not isinstance(cenwave, u.Quantity):
            cenwave = cenwave * self._internal_wave_unit

        bin_wave = units.validate_quantity(
            self.binset, cenwave.unit, equivalencies=u.spectral())

        return binning.wave_range(
            bin_wave.value, cenwave.value, npix, **kwargs) * cenwave.unit