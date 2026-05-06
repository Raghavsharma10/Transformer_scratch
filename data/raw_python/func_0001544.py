def _contiguous_slices(self):
        """Internal iterator over contiguous slices in RangeSet."""
        k = j = None
        for i in self._sorted():
            if k is None:
                k = j = i
            if i - j > 1:
                yield slice(k, j + 1, 1)
                k = i
            j = i
        if k is not None:
            yield slice(k, j + 1, 1)