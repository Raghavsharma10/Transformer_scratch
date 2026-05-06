def parse_value(self, value):
        """Cast value to `bool`."""
        parsed = super(BoolField, self).parse_value(value)
        return bool(parsed) if parsed is not None else None