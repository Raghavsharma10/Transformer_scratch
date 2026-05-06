def var(self):
        """
        Variance value as a result of an uncertainty calculation
        """
        mn = self.mean
        vr = np.mean((self._mcpts - mn) ** 2)
        return vr