def set_pixels(self, pixels):
        """
        Set the image data.
        Will not work if the new image has a different shape than the current image.

        Parameters
        ----------
        pixels : numpy.ndarray
            New image data

        Returns
        -------
        None
        """
        if not (pixels.shape == self._pixels.shape):
            raise AssertionError("Shape mismatch between pixels supplied {0} and existing image pixels {1}".format(pixels.shape,self._pixels.shape))
        self._pixels = pixels
        # reset this so that it is calculated next time the function is called
        self._rms = None
        return