def get_psf_sky(self, ra, dec):
        """
        Determine the local psf at a given sky location.
        The psf is returned in degrees.


        Parameters
        ----------
        ra, dec : float
            The sky position (degrees).

        Returns
        -------
        a, b, pa : float
            The psf semi-major axis, semi-minor axis, and position angle in (degrees).
            If a psf is defined then it is the psf that is returned, otherwise the image
            restoring beam is returned.
        """
        # If we don't have a psf map then we just fall back to using the beam
        # from the fits header (including ZA scaling)
        if self.data is None:
            beam = self.wcshelper.get_beam(ra, dec)
            return beam.a, beam.b, beam.pa

        x, y = self.sky2pix([ra, dec])
        # We leave the interpolation in the hands of whoever is making these images
        # clamping the x,y coords at the image boundaries just makes sense
        x = int(np.clip(x, 0, self.data.shape[1] - 1))
        y = int(np.clip(y, 0, self.data.shape[2] - 1))
        psf_sky = self.data[:, x, y]
        return psf_sky