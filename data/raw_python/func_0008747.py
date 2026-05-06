def union(self, other, renorm=True):
        """
        Add another Region by performing union on their pixlists.

        Parameters
        ----------
        other : :class:`AegeanTools.regions.Region`
            The region to be combined.

        renorm : bool
            Perform renormalisation after the operation?
            Default = True.
        """
        # merge the pixels that are common to both
        for d in range(1, min(self.maxdepth, other.maxdepth)+1):
            self.add_pixels(other.pixeldict[d], d)

        # if the other region is at higher resolution, then include a degraded version of the remaining pixels.
        if self.maxdepth < other.maxdepth:
            for d in range(self.maxdepth+1, other.maxdepth+1):
                for p in other.pixeldict[d]:
                    # promote this pixel to self.maxdepth
                    pp = p/4**(d-self.maxdepth)
                    self.pixeldict[self.maxdepth].add(pp)
        if renorm:
            self._renorm()
        return