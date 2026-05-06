def pix2sky_ellipse(self, pixel, sx, sy, theta):
        """
        Convert an ellipse from pixel to sky coordinates.

        Parameters
        ----------
        pixel : (float, float)
            The (x, y) coordinates of the center of the ellipse.
        sx, sy : float
            The major and minor axes (FHWM) of the ellipse, in pixels.
        theta : float
            The rotation angle of the ellipse (degrees).
            theta = 0 corresponds to the ellipse being aligned with the x-axis.

        Returns
        -------
        ra, dec : float
            The (ra, dec) coordinates of the center of the ellipse (degrees).

        a, b : float
            The semi-major and semi-minor axis of the ellipse (degrees).

        pa : float
            The position angle of the ellipse (degrees).
        """
        ra, dec = self.pix2sky(pixel)
        x, y = pixel
        v_sx = [x + sx * np.cos(np.radians(theta)),
                y + sx * np.sin(np.radians(theta))]
        ra2, dec2 = self.pix2sky(v_sx)
        major = gcd(ra, dec, ra2, dec2)
        pa = bear(ra, dec, ra2, dec2)

        v_sy = [x + sy * np.cos(np.radians(theta - 90)),
                y + sy * np.sin(np.radians(theta - 90))]
        ra2, dec2 = self.pix2sky(v_sy)
        minor = gcd(ra, dec, ra2, dec2)
        pa2 = bear(ra, dec, ra2, dec2) - 90

        # The a/b vectors are perpendicular in sky space, but not always in pixel space
        # so we have to account for this by calculating the angle between the two vectors
        # and modifying the minor axis length
        defect = pa - pa2
        minor *= abs(np.cos(np.radians(defect)))
        return ra, dec, major, minor, pa