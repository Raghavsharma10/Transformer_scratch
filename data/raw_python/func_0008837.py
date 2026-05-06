def pix2sky(self, pixel):
        """
        Get the sky coordinates for a given image pixel.

        Parameters
        ----------
        pixel : (float, float)
            Image coordinates.

        Returns
        -------
        ra,dec : float
            Sky coordinates (degrees)

        """
        pixbox = numpy.array([pixel, pixel])
        skybox = self.wcs.all_pix2world(pixbox, 1)
        return [float(skybox[0][0]), float(skybox[0][1])]