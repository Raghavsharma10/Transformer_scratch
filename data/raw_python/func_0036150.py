def encode(self, value):
        """Encode value."""
        value = self.serialize(value)
        if self.encoding:
            value = value.encode(self.encoding)
        return value