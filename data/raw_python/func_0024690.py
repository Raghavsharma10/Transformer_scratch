def waverange(self):
        """Range of `waveset`."""
        if self.waveset is None:
            x = [None, None]
        else:
            x = u.Quantity([self.waveset.min(), self.waveset.max()])
        return x