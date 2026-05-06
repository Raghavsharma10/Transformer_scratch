def add_pixels(self, pix, depth):
        """
        Add one or more HEALPix pixels to this region.

        Parameters
        ----------
        pix : int or iterable
            The pixels to be added

        depth : int
            The depth at which the pixels are added.
        """
        if depth not in self.pixeldict:
            self.pixeldict[depth] = set()
        self.pixeldict[depth].update(set(pix))