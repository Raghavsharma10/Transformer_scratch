def rmswidth(self, wavelengths=None, threshold=None):
        """Calculate the :ref:`bandpass RMS width <synphot-formula-rmswidth>`.
        Not to be confused with :func:`photbw`.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        threshold : float or `~astropy.units.quantity.Quantity`, optional
            Data points with throughput below this value are not
            included in the calculation. By default, all data points
            are included.

        Returns
        -------
        rms_width : `~astropy.units.quantity.Quantity`
            RMS width of the bandpass.

        Raises
        ------
        synphot.exceptions.SynphotError
            Threshold is invalid.

        """
        x = self._validate_wavelengths(wavelengths).value
        y = self(x).value

        if threshold is None:
            wave = x
            thru = y
        else:
            if (isinstance(threshold, numbers.Real) or
                (isinstance(threshold, u.Quantity) and
                 threshold.unit == self._internal_flux_unit)):
                mask = y >= threshold
            else:
                raise exceptions.SynphotError(
                    '{0} is not a valid threshold'.format(threshold))
            wave = x[mask]
            thru = y[mask]

        a = self.avgwave(wavelengths=wavelengths).value
        num = np.trapz((wave - a) ** 2 * thru, x=wave)
        den = np.trapz(thru, x=wave)

        if den == 0:  # pragma: no cover
            rms_width = 0.0
        else:
            rms_width = np.sqrt(abs(num / den))

        return rms_width * self._internal_wave_unit