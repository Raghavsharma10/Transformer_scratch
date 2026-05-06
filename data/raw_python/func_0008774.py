def get_pixbeam(self, ra, dec):
        """
        Get the psf at the location specified in pixel coordinates.
        The psf is also in pixel coordinates.

        Parameters
        ----------
        ra, dec : float
            The sky position (degrees).

        Returns
        -------
        a, b, pa : float
            The psf semi-major axis (pixels), semi-minor axis (pixels), and rotation angle (degrees).
            If a psf is defined then it is the psf that is returned, otherwise the image
            restoring beam is returned.

        """
        # If there is no psf image then just use the fits header (plus lat scaling) from the wcshelper
        if self.data is None:
            return self.wcshelper.get_pixbeam(ra, dec)
        # get the beam from the psf image data
        psf = self.get_psf_pix(ra, dec)
        if not np.all(np.isfinite(psf)):
            log.warn("PSF requested, returned Null")
            return None
        return Beam(psf[0], psf[1], psf[2])