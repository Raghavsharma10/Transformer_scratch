def effective_wavelength(self, binned=True, wavelengths=None,
                             mode='efflerg'):
        """Calculate :ref:`effective wavelength <synphot-formula-effwave>`.

        Parameters
        ----------
        binned : bool
            Sample data in native wavelengths if `False`.
            Else, sample binned data (default).

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` or `binset` is used, depending
            on ``binned``.

        mode : {'efflerg', 'efflphot'}
            Flux is first converted to the unit below before calculation:

                * 'efflerg' - FLAM
                * 'efflphot' - PHOTLAM (deprecated)

        Returns
        -------
        eff_lam : `~astropy.units.quantity.Quantity`
            Observation effective wavelength.

        Raises
        ------
        synphot.exceptions.SynphotError
            Invalid mode.

        """
        mode = mode.lower()
        if mode == 'efflerg':
            flux_unit = units.FLAM
        elif mode == 'efflphot':
            warnings.warn(
                'Usage of EFFLPHOT is deprecated.', AstropyDeprecationWarning)
            flux_unit = units.PHOTLAM
        else:
            raise exceptions.SynphotError(
                'mode must be "efflerg" or "efflphot"')

        if binned:
            x = self._validate_binned_wavelengths(wavelengths).value
            y = self.sample_binned(wavelengths=x, flux_unit=flux_unit).value
        else:
            x = self._validate_wavelengths(wavelengths).value
            y = units.convert_flux(x, self(x), flux_unit).value

        num = np.trapz(y * x ** 2, x=x)
        den = np.trapz(y * x, x=x)

        if den == 0.0:  # pragma: no cover
            eff_lam = 0.0
        else:
            eff_lam = abs(num / den)

        return eff_lam * self._internal_wave_unit