def wpeak(self, wavelengths=None):
        """Calculate
        :ref:`wavelength at peak throughput <synphot-formula-tpeak>`.

        If there are multiple data points with peak throughput
        value, only the first match is returned.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        Returns
        -------
        wpeak : `~astropy.units.quantity.Quantity`
            Wavelength at peak throughput.

        """
        x = self._validate_wavelengths(wavelengths)
        return x[self(x) == self.tpeak(wavelengths=wavelengths)][0]