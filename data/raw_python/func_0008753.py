def _uniq(self):
        """
        Create a list of all the pixels that cover this region.
        This list contains overlapping pixels of different orders.

        Returns
        -------
        pix : list
            A list of HEALPix pixel numbers.
        """
        pd = []
        for d in range(1, self.maxdepth):
            pd.extend(map(lambda x: int(4**(d+1) + x), self.pixeldict[d]))
        return sorted(pd)