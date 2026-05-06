def check_overlap(self, other, wavelengths=None, threshold=0.01):
        """Check for wavelength overlap between two spectra.

        Only wavelengths where ``self`` throughput is non-zero
        are considered.

        Example of full overlap::

            |---------- other ----------|
               |------ self ------|

        Examples of partial overlap::

            |---------- self ----------|
               |------ other ------|

            |---- other ----|
               |---- self ----|

            |---- self ----|
               |---- other ----|

        Examples of no overlap::

            |---- self ----|  |---- other ----|

            |---- other ----|  |---- self ----|

        Parameters
        ----------
        other : `BaseSpectrum`

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for integration.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, `waveset` is used.

        threshold : float
            If less than this fraction of flux or throughput falls
            outside wavelength overlap, the *lack* of overlap is
            *insignificant*. This is only used when partial overlap
            is detected. Default is 1%.

        Returns
        -------
        result : {'full', 'partial_most', 'partial_notmost', 'none'}
            * 'full' - ``self`` coverage is within or same as ``other``
            * 'partial_most' - Less than ``threshold`` fraction of
              ``self`` flux is outside the overlapping wavelength
              region, i.e., the *lack* of overlap is *insignificant*
            * 'partial_notmost' - ``self`` partially overlaps with
              ``other`` but does not qualify for 'partial_most'
            * 'none' - ``self`` does not overlap ``other``

        Raises
        ------
        synphot.exceptions.SynphotError
            Invalid inputs.

        """
        if not isinstance(other, BaseSpectrum):
            raise exceptions.SynphotError(
                'other must be spectrum or bandpass.')

        # Special cases where no sampling wavelengths given and
        # one of the inputs is continuous.
        if wavelengths is None:
            if other.waveset is None:
                return 'full'
            if self.waveset is None:
                return 'partial_notmost'

        x1 = self._validate_wavelengths(wavelengths)
        y1 = self(x1)
        a = x1[y1 > 0].value

        b = other._validate_wavelengths(wavelengths).value

        result = utils.overlap_status(a, b)

        if result == 'partial':
            # If there is no need to extrapolate or taper other
            # (i.e., other is zero at self's wave limits),
            # then we consider it as a full coverage.
            # This logic assumes __call__ never returns mag or count!
            if ((isinstance(other.model, Empirical1D) and
                 other.model.is_tapered() or
                 not isinstance(other.model,
                                (Empirical1D, _CompoundModel))) and
                    np.allclose(other(x1[::x1.size - 1]).value, 0)):
                result = 'full'

            # Check if the lack of overlap is significant.
            else:
                # Get all the flux
                totalflux = self.integrate(wavelengths=wavelengths).value
                utils.validate_totalflux(totalflux)

                a_min, a_max = a.min(), a.max()
                b_min, b_max = b.min(), b.max()

                # Now get the other two pieces
                excluded = 0.0
                if a_min < b_min:
                    excluded += self.integrate(
                        wavelengths=np.array([a_min, b_min])).value
                if a_max > b_max:
                    excluded += self.integrate(
                        wavelengths=np.array([b_max, a_max])).value

                if excluded / totalflux < threshold:
                    result = 'partial_most'
                else:
                    result = 'partial_notmost'

        return result