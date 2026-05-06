def extinction_curve(self, ebv, wavelengths=None):
        """Generate extinction curve.

        .. math::

            A(V) = R(V) \\; \\times \\; E(B-V)

            THRU = 10^{-0.4 \\; A(V)}

        Parameters
        ----------
        ebv : float or `~astropy.units.quantity.Quantity`
            :math:`E(B-V)` value in magnitude.

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        Returns
        -------
        extcurve : `ExtinctionCurve`
            Empirical extinction curve.

        Raises
        ------
        synphot.exceptions.SynphotError
            Invalid input.

        """
        if isinstance(ebv, u.Quantity) and ebv.unit.decompose() == u.mag:
            ebv = ebv.value
        elif not isinstance(ebv, numbers.Real):
            raise exceptions.SynphotError('E(B-V)={0} is invalid.'.format(ebv))

        x = self._validate_wavelengths(wavelengths).value
        y = 10 ** (-0.4 * self(x).value * ebv)
        header = {
            'E(B-V)': ebv,
            'ReddeningLaw': self.meta.get('expr', 'unknown')}

        return ExtinctionCurve(ExtinctionModel1D, points=x, lookup_table=y,
                               meta={'header': header})