def bins(start, end):
        """
        Get all the bin numbers for a particular interval defined by
        (start, end]
        """
        if end - start < 536870912:
            offsets = [585, 73, 9, 1]
        else:
            raise BigException
            offsets = [4681, 585, 73, 9, 1]
        binFirstShift = 17
        binNextShift = 3

        start = start >> binFirstShift
        end = (end - 1)  >> binFirstShift

        bins = [1]
        for offset in offsets:
            bins.extend(range(offset + start, offset + end + 1))
            start >>= binNextShift
            end >>= binNextShift
        return frozenset(bins)