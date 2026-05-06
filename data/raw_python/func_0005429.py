def guessbackgroundlevel(self):
        """
        Estimates the background level. This could be used to fill pixels in large cosmics.
        """
        if self.backgroundlevel == None:
            self.backgroundlevel = np.median(self.rawarray.ravel())
        return self.backgroundlevel