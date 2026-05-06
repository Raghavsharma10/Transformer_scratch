def plot(self, binned=True, wavelengths=None, flux_unit=None, area=None,
             vegaspec=None, **kwargs):  # pragma: no cover
        """Plot the observation.

        .. note:: Uses ``matplotlib``.

        Parameters
        ----------
        binned : bool
            Plot data in native wavelengths if `False`.
            Else, plot binned data (default).

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` or `binset` is used, depending
            on ``binned``.

        flux_unit : str or `~astropy.units.core.Unit` or `None`
            Flux is converted to this unit for plotting.
            If not given, internal unit is used.

        area, vegaspec
            See :func:`~synphot.units.convert_flux`.

        kwargs : dict
            See :func:`synphot.spectrum.BaseSpectrum.plot`.

        Raises
        ------
        synphot.exceptions.SynphotError
            Invalid inputs.

        """
        if binned:
            w, y = self._get_binned_arrays(wavelengths, flux_unit, area=area,
                                           vegaspec=vegaspec)
        else:
            w, y = self._get_arrays(wavelengths, flux_unit=flux_unit,
                                    area=area, vegaspec=vegaspec)
        self._do_plot(w, y, **kwargs)