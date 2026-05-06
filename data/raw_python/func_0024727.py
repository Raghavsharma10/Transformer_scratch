def sample_binned(self, wavelengths=None, flux_unit=None, **kwargs):
        """Sample binned observation without interpolation.

        To sample unbinned data, use ``__call__``.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, `binset` is used.

        flux_unit : str or `~astropy.units.core.Unit` or `None`
            Flux is converted to this unit.
            If not given, internal unit is used.

        kwargs : dict
            Keywords acceptable by :func:`~synphot.units.convert_flux`.

        Returns
        -------
        flux : `~astropy.units.quantity.Quantity`
            Binned flux in given unit.

        Raises
        ------
        synphot.exceptions.InterpolationNotAllowed
            Interpolation of binned data is not allowed.

        """
        x = self._validate_binned_wavelengths(wavelengths)
        i = np.searchsorted(self.binset, x)
        if not np.allclose(self.binset[i].value, x.value):
            raise exceptions.InterpolationNotAllowed(
                'Some or all wavelength values are not in binset.')
        y = self.binflux[i]

        if flux_unit is None:
            flux = y
        else:
            flux = units.convert_flux(x, y, flux_unit, **kwargs)

        return flux