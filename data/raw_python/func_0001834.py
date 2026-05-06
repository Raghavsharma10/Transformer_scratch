def format_value(self, value, padding):
        """Get padding adjusting for negative values."""

        # padding = padding - 1 if value < 0 and padding > 0 else padding
        # prefix = '-' if value < 0 else ''

        if padding:
            return "{:0{pad}d}".format(value, pad=padding)

        else:
            return str(value)