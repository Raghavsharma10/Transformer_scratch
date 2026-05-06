def get_area(self, degrees=True):
        """
        Calculate the total area represented by this region.

        Parameters
        ----------
        degrees : bool
            If True then return the area in square degrees, otherwise use steradians.
            Default = True.

        Returns
        -------
        area : float
            The area of the region.
        """
        area = 0
        for d in range(1, self.maxdepth+1):
            area += len(self.pixeldict[d])*hp.nside2pixarea(2**d, degrees=degrees)
        return area