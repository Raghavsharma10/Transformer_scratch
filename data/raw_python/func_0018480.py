def left(self, f, n=1):
        """return the nearest n features strictly to the left of a Feature f.
        Overlapping features are not considered as to the left.

        f: a Feature object
        n: the number of features to return
        """
        intervals = self.intervals[f.chrom]
        if intervals == []: return []

        iright = binsearch_left_start(intervals, f.start, 0 , len(intervals)) + 1
        ileft  = binsearch_left_start(intervals, f.start - self.max_len[f.chrom] - 1, 0, 0)

        results = sorted((distance(other, f), other) for other in intervals[ileft:iright] if other.end < f.start and distance(f, other) != 0)
        if len(results) == n:
            return [r[1] for r in results]

        # have to do some extra work here since intervals are sorted
        # by starts, and we dont know which end may be around...
        # in this case, we got some extras, just return as many as
        # needed once we see a gap in distances.
        for i in range(n, len(results)):
            if results[i - 1][0] != results[i][0]:
                return [r[1] for r in results[:i]]

        if ileft == 0:
            return [r[1] for r in results]

        # here, didn't get enough, so move left and try again. 
        1/0