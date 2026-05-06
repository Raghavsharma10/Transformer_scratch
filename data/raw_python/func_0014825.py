def to_python(self, value):
        """Converts any value to a base.Version field."""
        if value is None or value == '':
            return value
        if isinstance(value, base.Version):
            return value
        if self.coerce:
            return base.Version.coerce(value, partial=self.partial)
        else:
            return base.Version(value, partial=self.partial)