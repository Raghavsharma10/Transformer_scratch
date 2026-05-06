def _offset(self, x, y):
        """Helper for internal data"""
        x, y = force_int(x, y)
        return y * self.width * 4 + x * 4