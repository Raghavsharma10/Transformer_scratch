def tpeak(self, wavelengths=None):
        """Calculate :ref:`peak bandpass throughput <synphot-formula-tpeak>`.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        Returns
        -------
        tpeak : `~astropy.units.quantity.Quantity`
            Peak bandpass throughput.

        """
        x = self._validate_wavelengths(wavelengths).value
        return self(x).max()