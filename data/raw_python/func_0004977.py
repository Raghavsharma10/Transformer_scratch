def random(self: 'ErrorValue') -> np.ndarray:
        """Sample a random number (array) of the distribution defined by
        mean=`self.val` and variance=`self.err`^2.
        """
        if isinstance(self.val, np.ndarray):
            # IGNORE:E1103
            return np.random.randn(self.val.shape) * self.err + self.val
        else:
            return np.random.randn() * self.err + self.val