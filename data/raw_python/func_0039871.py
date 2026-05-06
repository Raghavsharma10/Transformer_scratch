def resample(self,N):
        """Returns a bootstrap resampling of provided samples.

        Parameters
        ----------
        N : int
            Number of samples.
        """
        inds = rand.randint(len(self.samples),size=N)
        return self.samples[inds]