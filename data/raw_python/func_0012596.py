def copy(self):
        """
        Returns a copy of the datamat.
        """
        return self.filter(np.ones(self._num_fix).astype(bool))