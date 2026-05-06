def contiguous(self):
        """Object-based iterator over contiguous range sets."""
        pad = self.padding or 0
        for sli in self._contiguous_slices():
            yield RangeSet.fromone(slice(sli.start, sli.stop, sli.step), pad)