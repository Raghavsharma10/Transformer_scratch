def apply(self, im):
        """
        Apply axis-localized displacements.

        Parameters
        ----------
        im : ndarray
            The image or volume to shift
        """
        from scipy.ndimage.interpolation import shift

        im = rollaxis(im, self.axis)
        im.setflags(write=True)
        for ind in range(0, im.shape[0]):
            im[ind] = shift(im[ind],  map(lambda x: -x, self.delta[ind]), mode='nearest')
        im = rollaxis(im, 0, self.axis+1)
        return im