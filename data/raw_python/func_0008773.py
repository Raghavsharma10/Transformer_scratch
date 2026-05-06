def get_psf_pix(self, ra, dec):
        """
        Determine the local psf (a,b,pa) at a given sky location.
        The psf is in pixel coordinates.

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
        psf_sky = self.get_psf_sky(ra, dec)
        psf_pix = self.wcshelper.sky2pix_ellipse([ra, dec], psf_sky[0], psf_sky[1], psf_sky[2])[2:]
        return psf_pix