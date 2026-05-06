def decode(self, value):
        """Decode value."""
        if self.encoding:
            value = value.decode(self.encoding)
        return self.deserialize(value)