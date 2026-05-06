def _blank_within(self, perimeter):
        """
        Blank all the pixels within the given perimeter.

        Parameters
        ----------
        perimeter : list
            The perimeter of the region.

        """
        # Method:
        # scan around the perimeter filling 'up' from each pixel
        # stopping when we reach the other boundary
        for p in perimeter:
            # if we are on the edge of the data then there is nothing to fill
            if p[0] >= self.data.shape[0] or p[1] >= self.data.shape[1]:
                continue
            # if this pixel is blank then don't fill
            if self.data[p] == 0:
                continue

            # blank this pixel
            self.data[p] = 0

            # blank until we reach the other perimeter
            for i in range(p[1]+1, self.data.shape[1]):
                q = p[0], i
                # stop when we reach another part of the perimeter
                if q in perimeter:
                    break
                # fill everything in between, even inclusions
                self.data[q] = 0

        return