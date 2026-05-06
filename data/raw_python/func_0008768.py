def get_pixbeam(self, ra, dec):
        """
        Determine the beam in pixels at the given location in sky coordinates.

        Parameters
        ----------
        ra , dec : float
            The sly coordinates at which the beam is determined.

        Returns
        -------
        beam : :class:`AegeanTools.fits_image.Beam`
            A beam object, with a/b/pa in pixel coordinates.
        """

        if ra is None:
            ra, dec = self.pix2sky(self.refpix)
        pos = [ra, dec]

        beam = self.get_beam(ra, dec)
        _, _, major, minor, theta = self.sky2pix_ellipse(pos, beam.a, beam.b, beam.pa)

        if major < minor:
            major, minor = minor, major
            theta -= 90
            if theta < -180:
                theta += 180
        if not np.isfinite(theta):
            theta = 0
        if not all(np.isfinite([major, minor, theta])):
            beam = None
        else:
            beam = Beam(major, minor, theta)
        return beam