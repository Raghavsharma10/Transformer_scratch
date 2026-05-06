def effstim(self, flux_unit=None, wavelengths=None, area=None,
                vegaspec=None):
        """Calculate :ref:`effective stimulus <synphot-formula-effstim>`
        for given flux unit.

        Parameters
        ----------
        flux_unit : str or `~astropy.units.core.Unit` or `None`
            The unit of effective stimulus.
            COUNT gives result in count/s (see :meth:`countrate` for more
            options).
            If not given, internal unit is used.

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling. This must be given if
            ``self.waveset`` is undefined for the underlying spectrum model(s).
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        area, vegaspec
            See :func:`~synphot.units.convert_flux`.

        Returns
        -------
        eff_stim : `~astropy.units.quantity.Quantity`
            Observation effective stimulus based on given flux unit.

        """
        if flux_unit is None:
            flux_unit = self._internal_flux_unit

        flux_unit = units.validate_unit(flux_unit)
        flux_unit_name = flux_unit.to_string()

        # Special handling of COUNT/OBMAG.
        # This is special case of countrate calculations.
        if flux_unit == u.count or flux_unit_name == units.OBMAG.to_string():
            val = self.countrate(area, binned=False, wavelengths=wavelengths)

            if flux_unit.decompose() == u.mag:
                eff_stim = (-2.5 * np.log10(val.value)) * flux_unit
            else:
                eff_stim = val

            return eff_stim

        # Special handling of VEGAMAG.
        # This is basically effstim(self)/effstim(Vega)
        if flux_unit_name == units.VEGAMAG.to_string():
            num = self.integrate(wavelengths=wavelengths)
            den = (vegaspec * self.bandpass).integrate()
            utils.validate_totalflux(num)
            utils.validate_totalflux(den)
            return (2.5 * (math.log10(den.value) -
                           math.log10(num.value))) * units.VEGAMAG

        # Sample the bandpass
        x_band = self.bandpass._validate_wavelengths(wavelengths).value
        y_band = self.bandpass(x_band).value

        # Sample the observation in FLAM
        inwave = self._validate_wavelengths(wavelengths).value
        influx = units.convert_flux(inwave, self(inwave), units.FLAM).value

        # Integrate
        num = abs(np.trapz(inwave * influx, x=inwave))
        den = abs(np.trapz(x_band * y_band, x=x_band))
        utils.validate_totalflux(num)
        utils.validate_totalflux(den)
        val = (num / den) * units.FLAM

        # Integration should always be done in FLAM and then
        # converted to desired units as follows.
        if flux_unit.physical_type == 'spectral flux density wav':
            if flux_unit == u.STmag:
                eff_stim = val.to(flux_unit)
            else:  # FLAM
                eff_stim = val
        elif flux_unit.physical_type in (
                'spectral flux density', 'photon flux density',
                'photon flux density wav'):
            w_pivot = self.bandpass.pivot()
            eff_stim = units.convert_flux(w_pivot, val, flux_unit)
        else:
            raise exceptions.SynphotError(
                'Flux unit {0} is invalid'.format(flux_unit))

        return eff_stim