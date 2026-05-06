def efficiency(self, wavelengths=None):
        """Calculate :ref:`dimensionless efficiency <synphot-formula-qtlam>`.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        Returns
        -------
        qtlam : `~astropy.units.quantity.Quantity`
            Dimensionless efficiency.

        """
        x = self._validate_wavelengths(wavelengths).value
        y = self(x).value
        qtlam = abs(np.trapz(y / x, x=x))
        return qtlam * u.dimensionless_unscaled