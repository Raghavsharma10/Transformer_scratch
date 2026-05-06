def vec2sky(cls, vec, degrees=False):
        """
        Convert [x,y,z] vectors into sky coordinates ra,dec

        Parameters
        ----------
        vec : numpy.array
            Unit vectors as an array of (x,y,z)

        degrees

        Returns
        -------
        sky : numpy.array
            Sky coordinates as an array of (ra,dec)

        See Also
        --------
        :func:`AegeanTools.regions.Region.sky2vec`
        """
        theta, phi = hp.vec2ang(vec)
        ra = phi
        dec = np.pi/2-theta

        if degrees:
            ra = np.degrees(ra)
            dec = np.degrees(dec)
        return cls.radec2sky(ra, dec)