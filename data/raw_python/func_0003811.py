def copy(self):
        """Return a deep copy"""
        result = Scalar(self.size, self.deriv)
        result.v = self.v
        if self.deriv > 0: result.d[:] = self.d[:]
        if self.deriv > 1: result.dd[:] = self.dd[:]
        return result