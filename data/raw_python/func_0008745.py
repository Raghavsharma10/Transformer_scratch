def _renorm(self):
        """
        Remake the pixel dictionary, merging groups of pixels at level N into a single pixel
        at level N-1
        """
        self.demoted = set()
        # convert all to lowest level
        self._demote_all()
        # now promote as needed
        for d in range(self.maxdepth, 2, -1):
            plist = self.pixeldict[d].copy()
            for p in plist:
                if p % 4 == 0:
                    nset = set((p, p+1, p+2, p+3))
                    if p+1 in plist and p+2 in plist and p+3 in plist:
                        # remove the four pixels from this level
                        self.pixeldict[d].difference_update(nset)
                        # add a new pixel to the next level up
                        self.pixeldict[d-1].add(p/4)
        self.demoted = set()
        return