def inv(self):
        """The inverse transformation"""
        result = Complete(self.r.transpose(), np.dot(self.r.transpose(), -self.t))
        result._cache_inv = self
        return result