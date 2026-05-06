def integrate(self, wavelengths=None, **kwargs):
        """Perform integration.

        This uses any analytical integral that the
        underlying model has (i.e., ``self.model.integral``).
        If unavailable, it uses the default fall-back integrator
        set in the ``default_integrator`` configuration item.

        If wavelengths are provided, flux or throughput is first resampled.
        This is useful when user wants to integrate at specific end points
        or use custom spacing; In that case, user can pass in desired
        sampling array generated with :func:`numpy.linspace`,
        :func:`numpy.logspace`, etc.
        If not provided, then `waveset` is used.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for integration.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, `waveset` is used.

        kwargs : dict
            Optional keywords to ``__call__`` for sampling.

        Returns
        -------
        result : `~astropy.units.quantity.Quantity`
            Integrated result.

        Raises
        ------
        NotImplementedError
            Invalid default integrator.

        synphot.exceptions.SynphotError
            `waveset` is needed but undefined or cannot integrate
            natively in the given ``flux_unit``.

        """
        # Cannot integrate per Hz units natively across wavelength
        # without converting them to per Angstrom unit first, so
        # less misleading to just disallow that option for now.
        if 'flux_unit' in kwargs:
            self._validate_flux_unit(kwargs['flux_unit'], wav_only=True)

        x = self._validate_wavelengths(wavelengths)

        # TODO: When astropy.modeling.models supports this, need to
        #       make sure that this actually works, and gives correct unit.
        # https://github.com/astropy/astropy/issues/5033
        # https://github.com/astropy/astropy/pull/5108
        try:
            m = self.model.integral
        except (AttributeError, NotImplementedError):
            if conf.default_integrator == 'trapezoid':
                y = self(x, **kwargs)
                result = abs(np.trapz(y.value, x=x.value))
                result_unit = y.unit
            else:  # pragma: no cover
                raise NotImplementedError(
                    'Analytic integral not available and default integrator '
                    '{0} is not supported'.format(conf.default_integrator))
        else:  # pragma: no cover
            start = x[0].value
            stop = x[-1].value
            result = (m(stop) - m(start))
            result_unit = self._internal_flux_unit

        # Ensure final unit takes account of integration across wavelength
        if result_unit != units.THROUGHPUT:
            if result_unit == units.PHOTLAM:
                result_unit = u.photon / (u.cm**2 * u.s)
            elif result_unit == units.FLAM:
                result_unit = u.erg / (u.cm**2 * u.s)
            else:  # pragma: no cover
                raise NotImplementedError(
                    'Integration of {0} is not supported'.format(result_unit))
        else:
            # Ideally flux can use this too but unfortunately this
            # operation results in confusing output unit for flux.
            result_unit *= self._internal_wave_unit

        return result * result_unit