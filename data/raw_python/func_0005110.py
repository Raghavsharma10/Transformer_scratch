def pilscale(self, r):
        """
        Converts a "scale" (like an aperture radius) of the original array or FITS file to the current PIL coordinates.
        """
        return r * float(self.upsamplefactor) / float(self.binfactor)