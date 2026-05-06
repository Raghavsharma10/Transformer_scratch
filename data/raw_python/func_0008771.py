def sky_sep(self, pix1, pix2):
        """
        calculate the GCD sky separation (degrees) between two pixels.

        Parameters
        ----------
        pix1, pix2 : (float, float)
            The (x,y) pixel coordinates for the two positions.

        Returns
        -------
        dist : float
            The distance between the two points (degrees).
        """
        pos1 = self.pix2sky(pix1)
        pos2 = self.pix2sky(pix2)
        sep = gcd(pos1[0], pos1[1], pos2[0], pos2[1])
        return sep