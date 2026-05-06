def isLearned(self, mode=None):
        """Return true if this parameter is learned

        Hidden parameters are not learned; automatic parameters inherit
        behavior from package/cl; other parameters are learned.
        If mode is set, it determines how automatic parameters behave.
        If not set, cl.mode parameter determines behavior.
        """
        if "l" in self.mode: return 1
        if "h" in self.mode: return 0
        if "a" in self.mode:
            if mode is None: mode = 'ql' # that is, iraf.cl.mode
            if "h" in mode and "l" not in mode:
                return 0
        return 1