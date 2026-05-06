def add_poly(self, positions, depth=None):
        """
        Add a single polygon to this region.

        Parameters
        ----------
        positions : [[ra, dec], ...]
            Positions for the vertices of the polygon. The polygon needs to be convex and non-intersecting.

        depth : int
            The deepth at which the polygon will be inserted.
        """
        if not (len(positions) >= 3): raise AssertionError("A minimum of three coordinate pairs are required")

        if depth is None or depth > self.maxdepth:
            depth = self.maxdepth

        ras, decs = np.array(list(zip(*positions)))
        sky = self.radec2sky(ras, decs)
        pix = hp.query_polygon(2**depth, self.sky2vec(sky), inclusive=True, nest=True)
        self.add_pixels(pix, depth)
        self._renorm()
        return