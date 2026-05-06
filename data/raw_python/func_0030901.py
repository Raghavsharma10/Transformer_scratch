def apply(self, im):
        """
        Apply an n-dimensional displacement by shifting an image or volume.

        Parameters
        ----------
        im : ndarray
            The image or volume to shift
        """
        from scipy.ndimage.interpolation import shift
        return shift(im, map(lambda x: -x, self.delta), mode='nearest')