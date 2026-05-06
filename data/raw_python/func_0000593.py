def parse_value(self, value):
        """Cast value to `int`, e.g. from string or long"""
        parsed = super(IntField, self).parse_value(value)
        if parsed is None:
            return parsed
        return int(parsed)