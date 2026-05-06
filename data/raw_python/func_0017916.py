def get_cusum(self, mag):
        """
        Return max - min of cumulative sum.

        Parameters
        ----------
        mag : array_like
            An array of magnitudes.

        Returns
        -------
        mm_cusum : float
            Max - min of cumulative sum.
        """

        c = np.cumsum(mag - self.weighted_mean) / len(mag) / self.weighted_std

        return np.max(c) - np.min(c)