def countrate(self, area, binned=True, wavelengths=None, waverange=None,
                  force=False):
        """Calculate :ref:`effective stimulus <synphot-formula-effstim>`
        in count/s.

        Parameters
        ----------
        area : float or `~astropy.units.quantity.Quantity`
            Area that flux covers. If not a Quantity, assumed to be in
            :math:`cm^{2}`.

        binned : bool
            Sample data in native wavelengths if `False`.
            Else, sample binned data (default).

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling. This must be given if
            ``self.waveset`` is undefined for the underlying spectrum model(s).
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` or `binset` is used, depending
            on ``binned``.

        waverange : tuple of float, Quantity, or `None`
            Lower and upper limits of the desired wavelength range.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, the full range is used.

        force : bool
            If a wavelength range is given, partial overlap raises
            an exception when this is `False` (default). Otherwise,
            it returns calculation for the overlapping region.
            Disjoint wavelength range raises an exception regardless.

        Returns
        -------
        count_rate : `~astropy.units.quantity.Quantity`
            Observation effective stimulus in count/s.

        Raises
        ------
        synphot.exceptions.DisjointError
            Wavelength range does not overlap with observation.

        synphot.exceptions.PartialOverlap
            Wavelength range only partially overlaps with observation.

        synphot.exceptions.SynphotError
            Calculation failed.

        """
        # Sample the observation
        if binned:
            x = self._validate_binned_wavelengths(wavelengths).value
            y = self.sample_binned(wavelengths=x, flux_unit=u.count,
                                   area=area).value
        else:
            x = self._validate_wavelengths(wavelengths).value
            y = units.convert_flux(x, self(x), u.count,
                                   area=area).value

        # Use entire wavelength range
        if waverange is None:
            influx = y

        # Use given wavelength range
        else:
            w = units.validate_quantity(waverange, self._internal_wave_unit,
                                        equivalencies=u.spectral()).value
            stat = utils.overlap_status(w, x)
            w1 = w.min()
            w2 = w.max()

            if stat == 'none':
                raise exceptions.DisjointError(
                    'Observation and wavelength range are disjoint.')
            elif 'partial' in stat:
                if force:
                    warnings.warn(
                        'Count rate calculated only for wavelengths in the '
                        'overlap between observation and given range.',
                        AstropyUserWarning)
                    w1 = max(w1, x.min())
                    w2 = min(w2, x.max())
                else:
                    raise exceptions.PartialOverlap(
                        'Observation and wavelength range do not fully '
                        'overlap. You may use force=True to force this '
                        'calculation anyway.')
            elif stat != 'full':  # pragma: no cover
                raise exceptions.SynphotError(
                    'Overlap result of {0} is unexpected'.format(stat))

            if binned:
                if wavelengths is None:
                    bin_edges = self.bin_edges.value
                else:
                    bin_edges = binning.calculate_bin_edges(x).value
                i1 = np.searchsorted(bin_edges, w1) - 1
                i2 = np.searchsorted(bin_edges, w2)
                influx = y[i1:i2]
            else:
                mask = ((x >= w1) & (x <= w2))
                influx = y[mask]

        val = math.fsum(influx)
        utils.validate_totalflux(val)

        return val * (u.count / u.s)