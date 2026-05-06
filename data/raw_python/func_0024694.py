def avgwave(self, wavelengths=None):
        """Calculate the :ref:`average wavelength <synphot-formula-avgwv>`.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, `waveset` is used.

        Returns
        -------
        avg_wave : `~astropy.units.quantity.Quantity`
            Average wavelength.

        """
        x = self._validate_wavelengths(wavelengths).value
        y = self(x).value
        num = np.trapz(y * x, x=x)
        den = np.trapz(y, x=x)

        if den == 0:  # pragma: no cover
            avg_wave = 0.0
        else:
            avg_wave = abs(num / den)

        return avg_wave * self._internal_wave_unit