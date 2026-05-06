def get_pixbeam_pixel(self, x, y):
        """
        Determine the beam in pixels at the given location in pixel coordinates.

        Parameters
        ----------
        x , y : float
            The pixel coordinates at which the beam is determined.

        Returns
        -------
        beam : :class:`AegeanTools.fits_image.Beam`
            A beam object, with a/b/pa in pixel coordinates.
        """
        ra, dec = self.pix2sky((x, y))
        return self.get_pixbeam(ra, dec)