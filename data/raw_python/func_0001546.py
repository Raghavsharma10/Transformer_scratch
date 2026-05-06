def split(self, nbr):
        """
        Split the rangeset into nbr sub-rangesets (at most). Each
        sub-rangeset will have the same number of elements more or
        less 1. Current rangeset remains unmodified. Returns an
        iterator.

        >>> RangeSet("1-5").split(3) 
        RangeSet("1-2")
        RangeSet("3-4")
        RangeSet("foo5")
        """
        assert(nbr > 0)

        # We put the same number of element in each sub-nodeset.
        slice_size = len(self) / nbr
        left = len(self) % nbr

        begin = 0
        for i in range(0, min(nbr, len(self))):
            length = slice_size + int(i < left)
            yield self[begin:begin + length]
            begin += length