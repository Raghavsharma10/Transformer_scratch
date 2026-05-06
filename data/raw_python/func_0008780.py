def solid(self, x, y):
        """
        Determine whether the pixel x,y is nonzero

        Parameters
        ----------
        x, y : int
            The pixel of interest.

        Returns
        -------
        solid : bool
            True if the pixel is not zero.
        """
        if not(0 <= x < self.xsize) or not(0 <= y < self.ysize):
            return False
        if self.data[x, y] == 0:
            return False
        return True