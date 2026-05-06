def emflx(self, area, wavelengths=None):
        """Calculate
        :ref:`equivalent monochromatic flux <synphot-formula-emflx>`.

        Parameters
        ----------
        area, wavelengths
            See :func:`unit_response`.

        Returns
        -------
        em_flux : `~astropy.units.quantity.Quantity`
            Equivalent monochromatic flux.

        """
        t_lambda = self.tlambda(wavelengths=wavelengths)

        if t_lambda == 0:  # pragma: no cover
            em_flux = 0.0 * units.FLAM
        else:
            uresp = self.unit_response(area, wavelengths=wavelengths)
            equvw = self.equivwidth(wavelengths=wavelengths).value
            em_flux = uresp * equvw / t_lambda

        return em_flux