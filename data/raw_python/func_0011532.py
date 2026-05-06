def unconsumed_ranges(self):
        """Return an IntervalTree of unconsumed ranges, of the format
        (start, end] with the end value not being included
        """
        res = IntervalTree()

        prev = None

        # normal iteration is not in a predictable order
        ranges = sorted([x for x in self.range_set], key=lambda x: x.begin)

        for rng in ranges:
            if prev is None:
                prev = rng
                continue
            res.add(Interval(prev.end, rng.begin))
            prev = rng
        
        # means we've seeked past the end
        if len(self.range_set[self.tell()]) != 1:
            res.add(Interval(prev.end, self.tell()))

        return res