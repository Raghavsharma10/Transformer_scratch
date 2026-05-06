def _transform_in(self):
        """Return array of coordinates that can be mapped by Transform
        classes."""
        return np.array([
            [self.left, self.bottom, 0, 1],
            [self.right, self.top, 0, 1]])