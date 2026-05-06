def barlam(self, wavelengths=None):
        """Calculate :ref:`mean log wavelength <synphot-formula-barlam>`.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, `waveset` is used.

        Returns
        -------
        bar_lam : `~astropy.units.quantity.Quantity`
            Mean log wavelength.

        """
        x = self._validate_wavelengths(wavelengths).value
        y = self(x).value
        num = np.trapz(y * np.log(x) / x, x=x)
        den = np.trapz(y / x, x=x)

        if num == 0 or den == 0:  # pragma: no cover
            bar_lam = 0.0
        else:
            bar_lam = np.exp(abs(num / den))

        return bar_lam * self._internal_wave_unit