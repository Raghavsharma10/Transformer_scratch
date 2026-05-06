def normalize(self, renorm_val, band=None, wavelengths=None, force=False,
                  area=None, vegaspec=None):
        """Renormalize the spectrum to the given Quantity and band.

        .. warning::

            Redshift attribute (``z``) is reset to 0 in the normalized
            spectrum even if ``self.z`` is non-zero.
            This is because the normalization simply adds a scale
            factor to the existing composite model.
            This is confusing but should not affect the flux sampling.

        Parameters
        ----------
        renorm_val : number or `~astropy.units.quantity.Quantity`
            Value to renormalize the spectrum to. If not a Quantity,
            assumed to be in internal unit.

        band : `SpectralElement`
           Bandpass to use in renormalization.

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for renormalization.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        force : bool
            By default (`False`), renormalization is only done
            when band wavelength limits are within ``self``
            or at least 99% of the flux is within the overlap.
            Set to `True` to force renormalization for partial overlap
            (this changes the underlying model of ``self`` to always
            extrapolate, if applicable).
            Disjoint bandpass raises an exception regardless.

        area, vegaspec
            See :func:`~synphot.units.convert_flux`.

        Returns
        -------
        sp : obj
            Renormalized spectrum.

        Raises
        ------
        synphot.exceptions.DisjointError
            Renormalization band does not overlap with ``self``.

        synphot.exceptions.PartialOverlap
            Renormalization band only partially overlaps with ``self``
            and significant amount of flux falls outside the overlap.

        synphot.exceptions.SynphotError
            Invalid inputs or calculation failed.

        """
        warndict = {}

        if band is None:
            sp = self

        else:
            if not isinstance(band, SpectralElement):
                raise exceptions.SynphotError('Invalid bandpass.')

            stat = band.check_overlap(self, wavelengths=wavelengths)

            if stat == 'none':
                raise exceptions.DisjointError(
                    'Spectrum and renormalization band are disjoint.')

            elif 'partial' in stat:
                if stat == 'partial_most':
                    warn_str = 'At least'
                elif stat == 'partial_notmost' and force:
                    warn_str = 'Less than'
                else:
                    raise exceptions.PartialOverlap(
                        'Spectrum and renormalization band do not fully '
                        'overlap. You may use force=True to force the '
                        'renormalization to proceed.')

                warn_str = (
                    'Spectrum is not defined everywhere in renormalization '
                    'bandpass. {0} 99% of the band throughput has '
                    'data. Spectrum will be').format(warn_str)

                if self.force_extrapolation():
                    warn_str = ('{0} extrapolated at constant '
                                'value.').format(warn_str)
                else:
                    warn_str = ('{0} evaluated outside pre-defined '
                                'waveset.').format(warn_str)

                warnings.warn(warn_str, AstropyUserWarning)
                warndict['PartialRenorm'] = warn_str

            elif stat != 'full':  # pragma: no cover
                raise exceptions.SynphotError(
                    'Overlap result of {0} is unexpected.'.format(stat))

            sp = self.__mul__(band)

        if not isinstance(renorm_val, u.Quantity):
            renorm_val = renorm_val * self._internal_flux_unit

        renorm_unit_name = renorm_val.unit.to_string()
        w = sp._validate_wavelengths(wavelengths)

        if (renorm_val.unit == u.count or
                renorm_unit_name == units.OBMAG.to_string()):
            # Special handling for non-density units
            flux_tmp = sp(w, flux_unit=u.count, area=area)
            totalflux = flux_tmp.sum().value
            stdflux = 1.0
        else:
            totalflux = sp.integrate(wavelengths=wavelengths).value

            # VEGAMAG
            if renorm_unit_name == units.VEGAMAG.to_string():
                if not isinstance(vegaspec, SourceSpectrum):
                    raise exceptions.SynphotError(
                        'Vega spectrum is missing.')
                stdspec = vegaspec

            # Magnitude flux-density units
            elif renorm_val.unit in (u.STmag, u.ABmag):
                stdspec = SourceSpectrum(
                    ConstFlux1D, amplitude=(0 * renorm_val.unit))

            # Linear flux-density units
            else:
                stdspec = SourceSpectrum(
                    ConstFlux1D, amplitude=(1 * renorm_val.unit))

            if band is None:
                # TODO: Cannot get this to agree with results
                # from using a very large box bandpass.
                # stdflux = stdspec.integrate(wavelengths=w).value
                raise NotImplementedError('Must provide a bandpass')
            else:
                up = stdspec * band
                stdflux = up.integrate(wavelengths=wavelengths).value

        utils.validate_totalflux(totalflux)

        # Renormalize in magnitudes
        if (renorm_val.unit.decompose() == u.mag or
                isinstance(renorm_val.unit, u.LogUnit)):
            const = renorm_val.value + (2.5 *
                                        np.log10(totalflux / stdflux))
            newsp = self.__mul__(10**(-0.4 * const))
        # Renormalize in linear flux units
        else:
            const = renorm_val.value * (stdflux / totalflux)
            newsp = self.__mul__(const)

        newsp.warnings = warndict
        return newsp