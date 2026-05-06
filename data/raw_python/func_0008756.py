def sky2vec(cls, sky):
        """
        Convert sky positions in to 3d-vectors on the unit sphere.

        Parameters
        ----------
        sky : numpy.array
            Sky coordinates as an array of (ra,dec)

        Returns
        -------
        vec : numpy.array
            Unit vectors as an array of (x,y,z)

        See Also
        --------
        :func:`AegeanTools.regions.Region.vec2sky`
        """
        theta_phi = cls.sky2ang(sky)
        theta, phi = map(np.array, list(zip(*theta_phi)))
        vec = hp.ang2vec(theta, phi)
        return vec