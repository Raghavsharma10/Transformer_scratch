def unit_response(self, area, wavelengths=None):
        """Calculate :ref:`unit response <synphot-formula-uresp>`
        of this bandpass.

        Parameters
        ----------
        area : float or `~astropy.units.quantity.Quantity`
            Area that flux covers. If not a Quantity, assumed to be in
            :math:`cm^{2}`.

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        Returns
        -------
        uresp : `~astropy.units.quantity.Quantity`
            Flux (in FLAM) of a star that produces a response of
            one photon per second in this bandpass.

        """
        a = units.validate_quantity(area, units.AREA)

        # Only correct if wavelengths are in Angstrom.
        x = self._validate_wavelengths(wavelengths).value

        y = self(x).value * x
        int_val = abs(np.trapz(y, x=x))
        uresp = units.HC / (a.cgs * int_val)

        return uresp.value * units.FLAM