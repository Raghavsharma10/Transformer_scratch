def sky2pix(self, pos):
        """
        Convert sky coordinates into pixel coordinates.

        Parameters
        ----------
        pos : (float, float)
            The (ra, dec) sky coordinates (degrees)

        Returns
        -------
        pixel : (float, float)
            The (x,y) pixel coordinates

        """
        pixel = self.wcs.wcs_world2pix([pos], 1)
        # wcs and pyfits have oposite ideas of x/y
        return [pixel[0][1], pixel[0][0]]