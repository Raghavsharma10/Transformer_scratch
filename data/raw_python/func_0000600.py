def parse_value(self, value):
        """Parse string into instance of `datetime`."""
        if isinstance(value, datetime.datetime):
            return value
        if value:
            return parse(value)
        else:
            return None