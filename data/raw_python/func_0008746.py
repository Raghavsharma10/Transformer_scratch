def sky_within(self, ra, dec, degin=False):
        """
        Test whether a sky position is within this region

        Parameters
        ----------
        ra, dec : float
            Sky position.

        degin : bool
            If True the ra/dec is interpreted as degrees, otherwise as radians.
            Default = False.

        Returns
        -------
        within : bool
            True if the given position is within one of the region's pixels.
        """
        sky = self.radec2sky(ra, dec)

        if degin:
            sky = np.radians(sky)

        theta_phi = self.sky2ang(sky)
        # Set values that are nan to be zero and record a mask
        mask = np.bitwise_not(np.logical_and.reduce(np.isfinite(theta_phi), axis=1))
        theta_phi[mask, :] = 0

        theta, phi = theta_phi.transpose()
        pix = hp.ang2pix(2**self.maxdepth, theta, phi, nest=True)
        pixelset = self.get_demoted()
        result = np.in1d(pix, list(pixelset))
        # apply the mask and set the shonky values to False
        result[mask] = False
        return result