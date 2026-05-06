def has_insert(self, shape):
        """Returns True if any of the inserts have the given shape."""
        for insert in self.inserts:
            if insert.shape == shape:
                return True
        return False