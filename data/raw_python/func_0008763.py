def pix2sky_vec(self, pixel, r, theta):
        """
        Given and input position and vector in pixel coordinates, calculate
        the equivalent position and vector in sky coordinates.

        Parameters
        ----------
        pixel : (int,int)
            origin of vector in pixel coordinates
        r : float
            magnitude of vector in pixels
        theta : float
            angle of vector in degrees

        Returns
        -------
        ra, dec : float
            The (ra, dec) of the origin point (degrees).
        r, pa : float
            The magnitude and position angle of the vector (degrees).
        """
        ra1, dec1 = self.pix2sky(pixel)
        x, y = pixel
        a = [x + r * np.cos(np.radians(theta)),
             y + r * np.sin(np.radians(theta))]
        locations = self.pix2sky(a)
        ra2, dec2 = locations
        a = gcd(ra1, dec1, ra2, dec2)
        pa = bear(ra1, dec1, ra2, dec2)
        return ra1, dec1, a, pa