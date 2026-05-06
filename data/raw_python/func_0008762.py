def sky2pix_vec(self, pos, r, pa):
        """
        Convert a vector from sky to pixel coords.
        The vector has a magnitude, angle, and an origin on the sky.

        Parameters
        ----------
        pos : (float, float)
            The (ra, dec) of the origin of the vector (degrees).

        r : float
            The magnitude or length of the vector (degrees).

        pa : float
            The position angle of the vector (degrees).

        Returns
        -------
        x, y : float
            The pixel coordinates of the origin.
        r, theta : float
            The magnitude (pixels) and angle (degrees) of the vector.

        """
        ra, dec = pos
        x, y = self.sky2pix(pos)
        a = translate(ra, dec, r, pa)
        locations = self.sky2pix(a)
        x_off, y_off = locations
        a = np.sqrt((x - x_off) ** 2 + (y - y_off) ** 2)
        theta = np.degrees(np.arctan2((y_off - y), (x_off - x)))
        return x, y, a, theta