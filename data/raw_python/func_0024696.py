def pivot(self, wavelengths=None):
        """Calculate :ref:`pivot wavelength <synphot-formula-pivwv>`.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, `waveset` is used.

        Returns
        -------
        pivwv : `~astropy.units.quantity.Quantity`
            Pivot wavelength.

        """
        x = self._validate_wavelengths(wavelengths).value
        y = self(x).value
        num = np.trapz(y * x, x=x)
        den = np.trapz(y / x, x=x)

        if den == 0:  # pragma: no cover
            pivwv = 0.0
        else:
            pivwv = np.sqrt(abs(num / den))

        return pivwv * self._internal_wave_unit