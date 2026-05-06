def get_int_range(self, start, end, increment=None):
        """Get an integer range between start and end and increments of increment."""

        first, last = int(start), int(end)
        increment = int(increment) if increment is not None else 1
        max_length = max(len(start), len(end))

        # Zero doesn't make sense as an incrementer
        # but like bash, just assume one
        if increment == 0:
            increment = 1

        if start[0] == '-':
            start = start[1:]

        if end[0] == '-':
            end = end[1:]

        if (len(start) > 1 and start[0] == '0') or (len(end) > 1 and end[0] == '0'):
            padding = max_length

        else:
            padding = 0

        if first < last:
            r = range(first, last + 1, -increment if increment < 0 else increment)

        else:
            r = range(first, last - 1, increment if increment < 0 else -increment)

        return (self.format_value(value, padding) for value in r)