def parse_value(self, value):
        """Parse string into instance of `time`."""
        if value is None:
            return value
        if isinstance(value, datetime.time):
            return value
        return parse(value).timetz()