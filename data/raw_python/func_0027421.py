def nonzero(self):
        """
        Get all non-zero bits
        """
        return [i for i in xrange(self.size()) if self.test(i)]