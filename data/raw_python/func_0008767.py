def get_beam(self, ra, dec):
        """
        Determine the beam at the given sky location.

        Parameters
        ----------
        ra, dec : float
            The sky coordinates at which the beam is determined.

        Returns
        -------
        beam : :class:`AegeanTools.fits_image.Beam`
            A beam object, with a/b/pa in sky coordinates
        """
        # check to see if we need to scale the major axis based on the declination
        if self.lat is None:
            factor = 1
        else:
            # this works if the pa is zero. For non-zero pa it's a little more difficult
            factor = np.cos(np.radians(dec - self.lat))
        return Beam(self.beam.a / factor, self.beam.b, self.beam.pa)