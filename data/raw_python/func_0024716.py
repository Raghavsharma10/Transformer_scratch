def fwhm(self, **kwargs):
        """Calculate :ref:`synphot-formula-fwhm` of equivalent gaussian.

        Parameters
        ----------
        kwargs : dict
            See :func:`photbw`.

        Returns
        -------
        fwhm_val : `~astropy.units.quantity.Quantity`
            FWHM of equivalent gaussian.

        """
        return np.sqrt(8 * np.log(2)) * self.photbw(**kwargs)