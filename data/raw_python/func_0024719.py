def rectwidth(self, wavelengths=None):
        """Calculate :ref:`bandpass rectangular width <synphot-formula-rectw>`.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        Returns
        -------
        rectw : `~astropy.units.quantity.Quantity`
            Bandpass rectangular width.

        """
        equvw = self.equivwidth(wavelengths=wavelengths)
        tpeak = self.tpeak(wavelengths=wavelengths)

        if tpeak.value == 0:  # pragma: no cover
            rectw = 0.0 * self._internal_wave_unit
        else:
            rectw = equvw / tpeak

        return rectw