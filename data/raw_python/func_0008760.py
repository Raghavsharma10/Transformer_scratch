def pix2sky(self, pixel):
        """
        Convert pixel coordinates into sky coordinates.

        Parameters
        ----------
        pixel : (float, float)
            The (x,y) pixel coordinates

        Returns
        -------
        sky : (float, float)
            The (ra,dec) sky coordinates in degrees

        """
        x, y = pixel
        # wcs and pyfits have oposite ideas of x/y
        return self.wcs.wcs_pix2world([[y, x]], 1)[0]