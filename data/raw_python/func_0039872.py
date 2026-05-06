def resample(self,N):
        """Returns a random sampling.
        """
        return rand.random(size=N)*(self.maxval - self.minval) + self.minval