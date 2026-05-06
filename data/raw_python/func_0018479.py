def find(self, start, end, chrom=None):
        """Return a object of all stored intervals intersecting between (start, end) inclusive."""
        intervals = self.intervals[chrom]
        ilen = len(intervals)
        # NOTE: we only search for starts, since any feature that starts within max_len of
        # the query could overlap, we must subtract max_len from the start to get the needed
        # search space. everything else proceeds like a binary search.
        # (but add distance calc for candidates).
        if not chrom in self.max_len: return []
        ileft  = binsearch_left_start(intervals, start - self.max_len[chrom], 0, ilen)
        iright = binsearch_right_end(intervals, end, ileft, ilen)
        query = Feature(start, end)
        # we have to check the distance to make sure we didnt pick up anything 
        # that started within max_len, but wasnt as long as max_len
        return [f for f in intervals[ileft:iright] if distance(f, query) == 0]