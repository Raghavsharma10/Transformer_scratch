def add_range(self, start, stop, step=1, pad=0):
        """
        Add a range (start, stop, step and padding length) to RangeSet.
        Like the Python built-in function range(), the last element is
        the largest start + i * step less than stop.
        """
        assert start < stop, "please provide ordered node index ranges"
        assert step > 0
        assert pad >= 0
        assert stop - start < 1e9, "range too large"

        if pad > 0 and self.padding is None:
            self.padding = pad
        set.update(self, range(start, stop, step))