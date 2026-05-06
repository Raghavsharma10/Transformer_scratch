def disable(self, everything=False):
        """
        Disable all but possibly not binning, which is needed for FF apps

        Parameters
        ---------
        everything : bool
            disable binning as well
        """
        self.freeze()
        if not everything:
            self.xbin.enable()
            self.ybin.enable()
        self.frozen = False