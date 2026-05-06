def sky2pix_ellipse(self, pos, a, b, pa):
        """
        Convert an ellipse from sky to pixel coordinates.

        Parameters
        ----------
        pos : (float, float)
            The (ra, dec) of the ellipse center (degrees).
        a, b, pa: float
            The semi-major axis, semi-minor axis and position angle of the ellipse (degrees).

        Returns
        -------
        x,y : float
            The (x, y) pixel coordinates of the ellipse center.
        sx, sy : float
            The major and minor axes (FWHM) in pixels.
        theta : float
            The rotation angle of the ellipse (degrees).
            theta = 0 corresponds to the ellipse being aligned with the x-axis.

        """
        ra, dec = pos
        x, y = self.sky2pix(pos)

        x_off, y_off = self.sky2pix(translate(ra, dec, a, pa))
        sx = np.hypot((x - x_off), (y - y_off))
        theta = np.arctan2((y_off - y), (x_off - x))

        x_off, y_off = self.sky2pix(translate(ra, dec, b, pa - 90))
        sy = np.hypot((x - x_off), (y - y_off))
        theta2 = np.arctan2((y_off - y), (x_off - x)) - np.pi / 2

        # The a/b vectors are perpendicular in sky space, but not always in pixel space
        # so we have to account for this by calculating the angle between the two vectors
        # and modifying the minor axis length
        defect = theta - theta2
        sy *= abs(np.cos(defect))

        return x, y, sx, sy, np.degrees(theta)