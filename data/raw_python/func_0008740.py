def add_circles(self, ra_cen, dec_cen, radius, depth=None):
        """
        Add one or more circles to this region

        Parameters
        ----------
        ra_cen, dec_cen, radius : float or list
            The center and radius of the circle or circles to add to this region.

        depth : int
            The depth at which the given circles will be inserted.

        """
        if depth is None or depth > self.maxdepth:
            depth = self.maxdepth
        try:
            sky = list(zip(ra_cen, dec_cen))
            rad = radius
        except TypeError:
            sky = [[ra_cen, dec_cen]]
            rad = [radius]
        sky = np.array(sky)
        rad = np.array(rad)
        vectors = self.sky2vec(sky)
        for vec, r in zip(vectors, rad):
            pix = hp.query_disc(2**depth, vec, r, inclusive=True, nest=True)
            self.add_pixels(pix, depth)
        self._renorm()
        return