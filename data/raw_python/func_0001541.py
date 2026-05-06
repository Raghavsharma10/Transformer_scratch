def striter(self):
        """Iterate over each (optionally padded) string element in RangeSet."""
        pad = self.padding or 0
        for i in self._sorted():
            yield "%0*d" % (pad, i)