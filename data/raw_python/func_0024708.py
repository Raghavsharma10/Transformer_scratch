def plot(self, wavelengths=None, flux_unit=None, area=None, vegaspec=None,
             **kwargs):  # pragma: no cover
        """Plot the spectrum.

        .. note:: Uses :mod:`matplotlib`.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for integration.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        flux_unit : str or `~astropy.units.core.Unit` or `None`
            Flux is converted to this unit for plotting.
            If not given, internal unit is used.

        area, vegaspec
            See :func:`~synphot.units.convert_flux`.

        kwargs : dict
            See :func:`BaseSpectrum.plot`.

        Raises
        ------
        synphot.exceptions.SynphotError
            Invalid inputs.

        """
        w, y = self._get_arrays(wavelengths, flux_unit=flux_unit, area=area,
                                vegaspec=vegaspec)
        self._do_plot(w, y, **kwargs)