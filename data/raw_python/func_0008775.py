def get_beam(self, ra, dec):
        """
        Get the psf as a :class:`AegeanTools.fits_image.Beam` object.

        Parameters
        ----------
        ra, dec : float
            The sky position (degrees).

        Returns
        -------
        beam : :class:`AegeanTools.fits_image.Beam`
            The psf at the given location.
        """
        if self.data is None:
            return self.wcshelper.beam
        else:
            psf = self.get_psf_sky(ra, dec)
            if not all(np.isfinite(psf)):
                return None
            return Beam(psf[0], psf[1], psf[2])