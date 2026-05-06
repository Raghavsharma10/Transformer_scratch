def _demote_all(self):
        """
        Convert the multi-depth pixeldict into a single set of pixels at the deepest layer.

        The result is cached, and reset when any changes are made to this region.
        """
        # only do the calculations if the demoted list is empty
        if len(self.demoted) == 0:
            pd = self.pixeldict
            for d in range(1, self.maxdepth):
                for p in pd[d]:
                    pd[d+1].update(set((4*p, 4*p+1, 4*p+2, 4*p+3)))
                pd[d] = set()  # clear the pixels from this level
            self.demoted = pd[d+1]
        return