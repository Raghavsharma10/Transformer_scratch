def inv(self):
        """The inverse rotation"""
        result = Rotation(self.r.transpose())
        result._cache_inv = self
        return result