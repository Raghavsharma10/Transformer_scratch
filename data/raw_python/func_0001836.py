def get_char_range(self, start, end, increment=None):
        """Get a range of alphabetic characters."""

        increment = int(increment) if increment else 1
        if increment < 0:
            increment = -increment

        # Zero doesn't make sense as an incrementer
        # but like bash, just assume one
        if increment == 0:
            increment = 1

        inverse = start > end
        alpha = _nalpha if inverse else _alpha

        start = alpha.index(start)
        end = alpha.index(end)

        if start < end:
            return (c for c in alpha[start:end + 1:increment])

        else:
            return (c for c in alpha[end:start + 1:increment])