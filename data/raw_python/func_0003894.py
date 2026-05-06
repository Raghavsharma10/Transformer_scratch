def inv(self):
        """The inverse translation"""
        result = Translation(-self.t)
        result._cache_inv = self
        return result